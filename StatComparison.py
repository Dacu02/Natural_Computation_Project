import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from typing import Literal
TEST_OPTIONS = Literal['Friedman', 'Friedman AR', 'Wilcoxon', 'auto']
AGGLOMERATION_METHOD = Literal['mean', 'median']
ALPHA_VALUE = 0.05

def CompareAlgorithms(data_path:str, skip_col:int|None=0, test_option:TEST_OPTIONS='auto', agglomeration_method:AGGLOMERATION_METHOD='mean'):
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified path '{data_path}' does not exist.")
    
    if os.path.isdir(data_path):
        all_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    else:
        all_files = [data_path.split('/')[-1]]
    dataframes = {}

    for file in all_files:
        csv_name = file.replace('.csv', '').split('_')[0]
        df = pd.read_csv(os.path.join(data_path, file))
        if skip_col is not None:
            df.drop(columns=[df.columns[skip_col]], inplace=True)
        dataframes[csv_name] = df

    experiments, algorithms = dataframes[next(iter(dataframes))].shape
    for df in dataframes.values():
        if df.shape != (experiments, algorithms):
            raise ValueError("All CSV files must have the same dimensions.")
    
    agg_func = np.mean if agglomeration_method == 'mean' else np.median
    results_df = pd.DataFrame(columns=['Rank', 'Algorithm'], index=range(algorithms))

    # for alg_name, df in dataframes.items():
    #     for col in df.columns:
    #         col_data = df[col]
    #         agg_value = agg_func(col_data)
    #         results_df.at[col, agglomeration_method] = agg_value

    # Da STAC
    if algorithms == 2:
        if test_option == 'auto':
            test_option = 'Wilcoxon'
        elif test_option != 'Wilcoxon':
            print("Attenzione, con due soli algoritmi è consigliato utilizzare il test di Wilcoxon.")
    elif algorithms > 2:
        if experiments < algorithms * 2:
            if test_option == 'auto':
                test_option = 'Friedman AR'
            elif test_option != 'Friedman AR':
                print("Attenzione, con più algoritmi e pochi esperimenti è consigliato utilizzare il test di Friedman con Average Ranks.")
        else:
            if test_option == 'auto':
                test_option = 'Friedman'
            elif test_option != 'Friedman':
                print("Attenzione, con più algoritmi e molti esperimenti è consigliato utilizzare il test di Friedman.")
    
    match test_option:
        case 'Friedman':
            for csv, df in dataframes.items():
                col_values = np.array([df[col].values for col in df.columns])
                stat, p_value = friedmanchisquare(*col_values)
                print(f"Friedman test for {csv}: statistic={stat}, p-value={p_value}")
                if not p_value < ALPHA_VALUE:
                    raise ValueError(f"Friedman test failed for {csv} with p-value: {p_value}")
                # ranks = rankdata(*[df[col] for col in df.columns], method='average') # Rank di Friedman
                # print(ranks)

        
        case _:
            raise ValueError("Invalid test_option provided.")
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        data_path = sys.argv[1]
        test_option = sys.argv[2]
    else:
        raise ValueError("Please provide the data path and test option as command line arguments.")
    
    CompareAlgorithms(data_path, skip_col=0, test_option=test_option) # type: ignore