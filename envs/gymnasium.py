import gymnasium as gym
import numpy as np
import cv2

class GymnasiumEnv(gym.Wrapper):
    def __init__(self, task, action_repeat=1, size=(64, 64), seed=0):
        env = gym.make(task, render_mode='rgb_array')
        self._action_repeat = action_repeat
        self._size = size
        super().__init__(env)
        self.env.reset(seed=seed)
    
    @property
    def observation_space(self):
        spaces = {}
        spaces['image'] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        spaces['state'] = self.env.observation_space
        return gym.spaces.Dict(spaces)
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def render(self):
        img = self.env.render()
        if img is None:
            raise ValueError("Render mode 'rgb_array' is not supported by the environment.")
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        return img
    
    def reset(self):
        obs, info = self.env.reset()
        obs_dict = {}
        obs_dict['image'] = self.render()
        obs_dict['state'] = obs
        obs_dict['is_terminal'] = False
        obs_dict['is_first'] = True
        return obs_dict

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            obs, r, term, trunc, info = self.env.step(action)
            reward += r
            if term or trunc:
                break
        obs_dict = {}
        obs_dict['state'] = obs
        obs_dict['image'] = self.render()
        obs_dict['is_terminal'] = term
        obs_dict['is_first'] = False
        return obs_dict, reward, term or trunc, info