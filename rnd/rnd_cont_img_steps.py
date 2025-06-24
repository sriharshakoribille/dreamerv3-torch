import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gymnasium as gym
from collections import deque
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from gymnasium.spaces import Box
import torchvision.transforms as T

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ActionRepeat(gym.Wrapper):

  def __init__(self, env, repeat):
    super().__init__(env)
    self._repeat = repeat

  def step(self, action):
    if action['reset']:
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    return obs
  
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        
        shape = env.observation_space.shape
        self.observation_space = Box(
            low=np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0),
            high=np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0),
            shape=(n_frames,) + shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        assert len(self.frames) == self.n_frames
        return np.stack(list(self.frames), axis=0)


class gae_trajectory_buffer(object):
    def __init__(self, capacity, gamma, lam):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.memory = deque(maxlen=self.capacity)
        # * [obs, next_obs, act, rew, don, val, ret, adv]

    def store(self, obs, next_obs, act, rew, don, val):
        obs = np.expand_dims(obs, 0)
        next_obs = np.expand_dims(next_obs, 0)
        self.memory.append([obs, next_obs, act, rew, don, val])

    def process(self):
        R = 0
        Adv = 0
        Value_previous = 0
        for traj in reversed(list(self.memory)):
            R = self.gamma * R * (1 - traj[4]) + traj[5]
            traj.append(R)
            # * the generalized advantage estimator(GAE)
            delta = traj[3] + Value_previous * self.gamma * (1 - traj[4]) - traj[5]
            Adv = delta + (1 - traj[4]) * Adv * self.gamma * self.lam
            traj.append(Adv)
            Value_previous = traj[5]

    def get(self):
        obs, next_obs, act, rew, don, val, ret, adv = zip(* self.memory)
        act = np.array(act, dtype=np.float32)
        rew = np.expand_dims(rew, 1)
        don = np.expand_dims(don, 1)
        val = np.expand_dims(val, 1)
        ret = np.expand_dims(ret, 1)
        adv = np.array(adv)
        adv = (adv - adv.mean()) / adv.std()
        adv = np.expand_dims(adv, 1)
        return np.concatenate(obs, 0), np.concatenate(next_obs, 0), act, rew, don, val, ret, adv

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()


class policy_net(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(policy_net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(512, output_dim)
        self.log_std = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, input):
        if input.max() > 1.0:
            input = input / 255.0
        conv_out = self.conv(input)
        x = self.fc(conv_out)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std

    def act(self, input):
        mu, std = self.forward(input)
        dist = Normal(mu, std)
        action = dist.sample()
        return action.detach().cpu().numpy().flatten()


class value_net(nn.Module):
    def __init__(self, input_shape):
        super(value_net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.int_layer = nn.Linear(512, 1)
        self.ext_layer = nn.Linear(512, 1)

    def forward(self, input):
        if input.max() > 1.0:
            input = input / 255.0
        conv_out = self.conv(input)
        x = self.fc(conv_out)
        value_int = self.int_layer(x)
        value_ext = self.ext_layer(x)
        return value_int, value_ext


class rnd(nn.Module):
    def __init__(self, input_shape):
        super(rnd, self).__init__()
        
        self.predictor = self._build_net(input_shape)
        self.target = self._build_net(input_shape)

        for param in self.target.parameters():
            param.requires_grad = False

    def _build_net(self, input_shape):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = conv(dummy_input).shape[1]

        net = nn.Sequential(
            conv,
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        return net

    def forward(self, input):
        if input.max() > 1.0:
            input = input / 255.0
        pre_feature = self.predictor(input)
        tar_feature = self.target(input)
        return pre_feature, tar_feature

    def calc_int_reward(self, input):
        pre_feature, tar_feature = self.forward(input)
        int_reward = 0.5 * (pre_feature - tar_feature).pow(2).sum(-1)
        return int_reward.detach().cpu().numpy()

class ppo_clip(object):
    def __init__(self, env, max_steps, learning_rate, gamma, lam, epsilon, 
                 capacity, render, log, update_iterations, int_coef, ext_coef, 
                 rnd_update_prop, seed, device, tb_path='runs_rnd/idk', 
                 eval_env=None, eval_freq=100):
        super(ppo_clip, self).__init__()
        self.env = env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.capacity = capacity
        self.render = render
        self.log = log
        self.update_iterations = update_iterations
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.rnd_update_prop = rnd_update_prop
        self.seed = seed
        self.device = device

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.observation_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low

        self.policy_net = policy_net(self.observation_dim, self.action_dim).to(self.device)
        self.value_net = value_net(self.observation_dim).to(self.device)
        self.rnd = rnd(self.observation_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=self.learning_rate)
        self.int_buffer = gae_trajectory_buffer(capacity=self.capacity, gamma=self.gamma, lam=self.lam)
        self.ext_buffer = gae_trajectory_buffer(capacity=self.capacity, gamma=self.gamma, lam=self.lam)

        self.count = 0
        self.train_count = 0
        self.weight_reward = None
        self.writer = SummaryWriter(tb_path)

    def train(self):
        obs, next_obs, act, int_rew, don, _, _, int_adv = self.int_buffer.get()
        _, _, _, ext_rew, _, _, _, ext_adv = self.ext_buffer.get()

        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        int_rew = torch.FloatTensor(int_rew).to(self.device)
        ext_rew = torch.FloatTensor(ext_rew).to(self.device)
        don = torch.FloatTensor(don).to(self.device)
        int_adv = torch.FloatTensor(int_adv).to(self.device)
        ext_adv = torch.FloatTensor(ext_adv).to(self.device)
        adv = self.int_coef * int_adv + self.ext_coef * ext_adv

        mu, std = self.policy_net.forward(obs)
        dist = Normal(mu, std)
        old_log_probs = dist.log_prob(act).sum(dim=1, keepdim=True).detach()

        value_loss_buffer = []
        policy_loss_buffer = []
        rnd_loss_buffer = []
        for _ in range(self.update_iterations):
            value_int, value_ext = self.value_net.forward(obs)
            next_value_int, next_value_ext = self.value_net.forward(next_obs)
            # * intrinsic value net
            int_td_target = int_rew + self.gamma * next_value_int * (1 - don)
            int_value_loss = F.mse_loss(int_td_target.detach(), value_int)
            # * external value net
            ext_td_target = ext_rew + self.gamma * next_value_ext * (1 - don)
            ext_value_loss = F.mse_loss(ext_td_target.detach(), value_ext)
            value_loss = 0.5 * (int_value_loss + ext_value_loss)

            value_loss_buffer.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            mu, std = self.policy_net.forward(obs)
            dist = Normal(mu, std)
            log_probs = dist.log_prob(act).sum(dim=1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv
            policy_loss = - torch.min(surr1, surr2).mean()
            policy_loss_buffer.append(policy_loss.item())
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            pre_feature, tar_feature = self.rnd.forward(obs)
            rnd_loss = (pre_feature - tar_feature.detach()).pow(2).mean(-1)
            mask = torch.rand(len(rnd_loss), device=self.device)
            mask = (mask < self.rnd_update_prop).float()
            rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1.], device=self.device))
            rnd_loss_buffer.append(rnd_loss.item())
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()
        if self.log:
            self.writer.add_scalar('rnd_loss', np.mean(rnd_loss_buffer), self.train_count)
            self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)
            self.writer.add_scalar('value_loss', np.mean(value_loss_buffer), self.train_count)

    def evaluate(self, step_num):
        self.policy_net.eval()
        total_reward = 0
        obs, _ = self.eval_env.reset()
        while True:
            with torch.no_grad():
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device))
            clipped_action = np.clip(action, self.action_low, self.action_high)
            next_obs, reward, terminated, truncated, _ = self.eval_env.step(clipped_action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
            if done:
                break
        
        if self.log:
            self.writer.add_scalar('eval_reward', total_reward, step_num)

        print('step: {}  evaluation reward: {:.2f}'.format(step_num, total_reward))
        self.policy_net.train()

    def run(self):
        total_steps = 0
        episode = 0
        last_eval_step = 0
        while total_steps < self.max_steps:
            if self.eval_env is not None and (total_steps - last_eval_step) >= self.eval_freq:
                self.evaluate(total_steps)
                last_eval_step = total_steps
            obs, _ = self.env.reset(seed=self.seed + episode)
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device))
                clipped_action = np.clip(action, self.action_low, self.action_high)
                next_obs, ext_reward, terminated, truncated, _ = self.env.step(clipped_action)
                done = terminated or truncated
                int_reward = self.rnd.calc_int_reward(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device))[0]
                if self.render:
                    self.env.render()
                value_int, value_ext = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device))
                value_int = value_int.detach().item()
                value_ext = value_ext.detach().item()
                self.ext_buffer.store(obs, next_obs, action, ext_reward, done, value_ext)
                self.int_buffer.store(obs, next_obs, action, int_reward, done, value_int)
                self.count += 1
                total_reward += ext_reward
                obs = next_obs
                total_steps += 1
                if self.count % self.capacity == 0:
                    self.int_buffer.process()
                    self.ext_buffer.process()
                    self.train_count += 1
                    self.train()
                    self.int_buffer.clear()
                    self.ext_buffer.clear()
                if done or total_steps >= self.max_steps:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = self.weight_reward * 0.99 + total_reward * 0.01
                    if self.log:
                        self.writer.add_scalar('weight_reward', self.weight_reward, total_steps)
                        self.writer.add_scalar('reward', total_reward, total_steps)
                    print('step: {}  episode: {}  reward: {:.2f}  weight_reward: {:.2f}  train_step: {}'.format(
                        total_steps, episode + 1, total_reward, self.weight_reward, self.train_count))
                    break
            episode += 1

# def make_env(env_name, seed=None):
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v5', help='Gym environment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    env_name = args.env_name
    seed = args.seed
    # env_name = 'Hopper-v5'
    env = gym.make(env_name, render_mode='rgb_array')
    _,_ = env.reset(seed=0)
    env = gym.wrappers.AddRenderObservation(env,render_only=True)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 64)
    env = FrameStack(env, 4)

    eval_env = gym.make(env_name, render_mode='rgb_array')
    _,_ = eval_env.reset(seed=0)
    eval_env = gym.wrappers.AddRenderObservation(eval_env, render_only=True)
    eval_env = gym.wrappers.RenderCollection(eval_env)
    eval_env = GrayScaleObservation(eval_env)
    eval_env = ResizeObservation(eval_env, 64)
    eval_env = FrameStack(eval_env, 4)
    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    test = ppo_clip(
        env=env,
        eval_env=eval_env,
        max_steps=1_000_000,  # <-- changed from 'episode' to 'max_steps'
        learning_rate=1e-4,
        gamma=0.99,
        lam=0.97,
        epsilon=0.2,
        capacity=2048,
        render=False,
        log=True,
        update_iterations=10,
        int_coef=1.,
        ext_coef=2.,
        rnd_update_prop=0.25,
        seed=seed,
        device=device,
        eval_freq=15_000,  # <-- evaluate every 25k steps
        # tb_path='runs_rnd/ppo_clip_rnd/gym/cheetah_run/vision/s0'
        tb_path=f'runs_rnd/ppo_clip_rnd/gym/{env_name}/vision/s{seed}',
        # tb_path='runs_rnd/debug'
    )
    test.run()