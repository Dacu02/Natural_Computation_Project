from time import strftime
import pandas as pd
import os
def main():
    results_path = os.path.join(os.getcwd(), 'results')
    to_merge = {}
    output_path = os.path.join(os.getcwd(), 'aggregated_results')
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'aggregated_results__{strftime("%d__%H_%M")}')
    os.makedirs(output_path, exist_ok=True)

    runned_once = False
    for problem_folder in os.listdir(results_path):
        if os.path.isdir(os.path.join(results_path, problem_folder)):
            for file in os.listdir(os.path.join(results_path, problem_folder)):
                 if file.endswith('.csv'):
                    problem_id = int(file.split('_')[0])
                    if not runned_once and problem_id not in to_merge.keys():
                        to_merge[problem_id] = []
                    elif runned_once and problem_id not in to_merge.keys():
                        raise ValueError("Inconsistent problem IDs across runs.")
                    to_merge[problem_id].append(os.path.join(results_path, problem_folder, file))
        runned_once = True

    for problem_id, files in sorted(to_merge.items()):
        merged_df = pd.DataFrame()
        for file in files:
            df = pd.read_csv(os.path.join(results_path, file))
            merged_df = pd.concat([merged_df, df], ignore_index=False, axis=0)
        merged_df.to_csv(os.path.join(output_path, f'{problem_id}_summary_results.csv'), index=False)

    os.makedirs(os.path.join(os.getcwd(), 'aggregated_results'), exist_ok=True)
    
if __name__ == "__main__":
    main()