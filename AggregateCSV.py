import sys
from time import strftime
import pandas as pd
import os
from functools import reduce
def retrieve_class(problem_number: int):
   return (problem_number - 1) // 8 + 1

def main():
    if len(sys.argv) > 1:
        name_file = sys.argv[1]
    else:
        raise ValueError("Please provide the name of the file to aggregate as a command-line argument.")
    
    results_path = os.path.join(os.getcwd(), 'results')
    to_merge:dict[int, list[str]] = {}
    output_path = os.path.join(os.getcwd(), 'aggregated_results')
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'aggregated_results__{name_file}')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'by_problem'), exist_ok=True)

    alg_parameters = []
    seeds = []
    runned_once = False
    folders_to_visit = [os.path.join(results_path, folder) 
                        for folder in os.listdir(results_path) 
                            if os.path.isdir(os.path.join(results_path, folder)) 
                                and folder.endswith(name_file)]
    for problem_folder in folders_to_visit:
        for file in os.listdir(problem_folder):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(problem_folder, file))
                if not runned_once and not alg_parameters:
                    alg_parameters = [col for col in df.columns if not col.strip().isdigit()]
                    seeds = [col for col in df.columns if col.strip().isdigit()]
                else:
                    current_parameters = [col for col in df.columns if not col.strip().isdigit()]
                    current_seeds = [col for col in df.columns if col.strip().isdigit()]
                    if set(current_parameters) != set(alg_parameters) or set(current_seeds) != set(seeds):
                        raise ValueError("Inconsistent algorithm parameters or seeds across runs.")
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
        merged_df.to_csv(os.path.join(output_path, 'by_problem', f'{problem_id}_summary_results.csv'), index=False)

    os.makedirs(os.path.join(output_path, 'by_class'), exist_ok=True)
    for class_id in [1, 2, 3]:
        class_df = []
        for summary_folder in os.listdir(os.path.join(output_path, 'by_problem')):
            problem_id = int(summary_folder.split('_')[0])
            if summary_folder.endswith('.csv') and retrieve_class(problem_id) == class_id:
                df = pd.read_csv(os.path.join(output_path, 'by_problem', summary_folder))
                rename_map = {col: f"{problem_id}_{col}" for col in seeds}
                df = df.rename(columns=rename_map)
                class_df.append(df)
        
        if len(class_df) == 0:
            print(f"No data found for class {class_id}. Skipping.")
            continue
        
        final_class_df = reduce(
            lambda left, right: pd.merge(left, right, on=alg_parameters, how='outer'), # type: ignore
            class_df
        )
        
        result_cols = [col for col in final_class_df.columns if col not in alg_parameters]
        result_cols.sort() 
        ordered_columns = alg_parameters + result_cols

        final_class_df = final_class_df[ordered_columns]
        final_class_df.to_csv(os.path.join(output_path, 'by_class', f'{class_id}_class_summary_results.csv'), index=False) # type: ignore
    

    os.makedirs(os.path.join(output_path, 'all'), exist_ok=True)
    all_problems = []
    for summary_folder in os.listdir(os.path.join(output_path, 'by_problem')):
        problem_id = int(summary_folder.split('_')[0])
        if summary_folder.endswith('.csv'):
            df = pd.read_csv(os.path.join(output_path, 'by_problem', summary_folder))
            columns = df.columns
            seeds_df = [col for col in columns if col.strip().isdigit()]
            params_df = [col for col in columns if not col.strip().isdigit()]
            if not set(params_df) == set(alg_parameters) or not set(seeds_df) == set(seeds):
                print(f"Warning: Inconsistent columns in file {summary_folder}. Skipping global aggregation")
                break
            rename_map = {col: f"{problem_id}_{col}" for col in seeds}
            df = df.rename(columns=rename_map)
            all_problems.append(df)

    if len(all_problems) != 0:
        all_problems_df = reduce(
            lambda left, right: pd.merge(left, right, on=alg_parameters, how='outer'), # type: ignore
            all_problems
        )
        all_result_cols = [col for col in all_problems_df.columns if col not in alg_parameters]
        all_result_cols.sort()
        ordered_columns = alg_parameters + all_result_cols
        all_problems_df = all_problems_df[ordered_columns]
        all_problems_df.to_csv(os.path.join(output_path, 'all', f'all_problems_summary_results.csv'), index=False)

                        

if __name__ == "__main__":
    main()