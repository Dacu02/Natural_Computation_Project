from time import strftime
import pandas as pd
import os
def main():
    results_path = os.path.join(os.getcwd(), 'results')
    to_merge = []
    for file in os.listdir(results_path):
        if file.endswith('.csv') and file.startswith('partial_run'):
            to_merge.append(file)

    merged_df = pd.DataFrame()
    for file in to_merge:
        df = pd.read_csv(os.path.join(results_path, file))
        merged_df = pd.concat([merged_df, df], ignore_index=False, axis=1)

    os.makedirs(os.path.join(os.getcwd(), 'aggregated_results'), exist_ok=True)
    merged_df.to_csv(os.path.join(os.getcwd(), 'aggregated_results', f'aggregated_results{strftime('%d__%H_%M')}.csv'), index=False)
if __name__ == "__main__":
    main()