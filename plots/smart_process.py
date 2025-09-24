import pandas as pd
import json
import argparse
import numpy as np
import os
import re

def process_and_bin_scores(input_file, output_file, group_threshold=1000):
    """
    Reads a JSONL file, automatically detects clusters of data points,
    calculates their mean, and preserves outlier/evaluation points.
    """
    try:
        # 1. Load data into a pandas DataFrame
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)

        if df.empty or 'step' not in df.columns:
            print(f"File {input_file} is empty or missing 'step' column.")
            return

        # Find the score column automatically
        score_col = next((col for col in ['episode/score', 'epstats/score'] if col in df.columns), None)
        if not score_col:
            print(f"Could not find a score column in {input_file}.")
            return
        df = df.rename(columns={score_col: 'score'})
        
        # 2. Sort by step and identify groups based on step difference
        df = df.sort_values('step').reset_index(drop=True)
        
        # A new group starts whenever the step difference is larger than the threshold
        df['group_id'] = (df['step'].diff() > group_threshold).cumsum()

        processed_data = []

        # 3. Iterate over each identified group to process it
        for _, group_df in df.groupby('group_id'):
            if len(group_df) > 1:
                # This is a cluster of training data points.
                # Calculate mean step, mean score, std, and count.
                processed_data.append({
                    "step": int(group_df['step'].mean()),
                    "score": group_df['score'].mean(),
                    "score_std": group_df['score'].std(),
                    "count": len(group_df),
                    "type": "binned_mean"
                })
            else:
                # This is a single point (outlier or evaluation score).
                # Keep it as is.
                point = group_df.iloc[0]
                processed_data.append({
                    "step": int(point['step']),
                    "score": point['score'],
                    "score_std": np.nan,
                    "count": 1,
                    "type": "single_point"
                })

        # 4. Create a new DataFrame and save to a new JSONL file
        output_df = pd.DataFrame(processed_data).sort_values('step')
        output_df.to_json(output_file, orient='records', lines=True)

        print(f"Successfully processed {input_file}.")
        print(f"Grouped and binned data saved to {output_file}.")

        # Bonus: Estimate the binning interval from the data
        large_jumps = df['step'].diff().dropna().loc[lambda x: x > group_threshold]
        if not large_jumps.empty:
            print(f"Estimated logging interval from data: ~{int(large_jumps.median())} steps.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recursively find and process RL score logs in a directory. '
                    'Detects data clusters, calculates their mean, and preserves single evaluation points.'
    )
    parser.add_argument('input_dir', type=str, help='The root directory to search for log files.')
    parser.add_argument(
        '--threshold', type=int, default=500,
        help='The step difference threshold to identify a new data cluster.'
    )
    
    args = parser.parse_args()

    # Regex to find files like s0_scores.jsonl, s1_scores.jsonl, etc.
    file_pattern = re.compile(r's\d+_scores\.jsonl')
    
    print(f"Starting search in directory: {args.input_dir}")
    found_files = 0

    for root, _, files in os.walk(args.input_dir):
        for filename in files:
            if file_pattern.match(filename):
                found_files += 1
                input_path = os.path.join(root, filename)
                
                # Create the new output filename, e.g., s0_scores.jsonl -> s0_scores_all.jsonl
                output_filename = filename.replace('_scores.jsonl', '_scores_all.jsonl')
                output_path = os.path.join(root, output_filename)
                
                print(f"\n--- Processing file: {input_path} ---")
                process_and_bin_scores(input_path, output_path, args.threshold)

    if found_files == 0:
        print("No matching files (e.g., s0_scores.jsonl) were found.")
    else:
        print(f"\nFinished processing. Found and processed {found_files} files.")