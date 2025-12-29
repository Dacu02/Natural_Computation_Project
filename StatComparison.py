import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from typing import Literal
ANALYSIS_PATH = os.path.join(os.getcwd(), 'comparison')
RANK_OPTIONS = Literal['Friedman', 'Friedman AR', 'Wilcoxon', 'auto']
AGGLOMERATION_METHOD = Literal['mean', 'median']
ALPHA_VALUE = 0.05
def CompareAlgorithms(data_path:str, skip_col:int=1, rank_option:RANK_OPTIONS='auto', agglomeration_method:AGGLOMERATION_METHOD='mean'):
    all_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not all_files:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The specified path '{data_path}' does not exist.")
        else:
            all_files = [data_path]
    dataframes = {}

    for file in all_files:
        csv_name = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_path, file))
        df.drop(columns=0)
        dataframes[csv_name] = df

    experiments, algorithms = dataframes[next(iter(dataframes))].shape
    for df in dataframes.values():
        if df.shape != (experiments, algorithms):
            raise ValueError("All CSV files must have the same dimensions.")
    
    agg_func = np.mean if agglomeration_method == 'mean' else np.median
    results_df = pd.DataFrame(columns=[agglomeration_method, 'Rank', 'Algorithm'], index=range(algorithms))


    for alg_name, df in dataframes.items():
        for col in df.columns:
            col_data = df[col]
            agg_value = agg_func(col_data)
            results_df.at[col, agglomeration_method] = agg_value

    # Da STAC
    if algorithms == 2:
        if rank_option == 'auto':
            rank_option = 'Wilcoxon'
        elif rank_option != 'Wilcoxon':
            print("Attenzione, con due soli algoritmi è consigliato utilizzare il test di Wilcoxon.")
    elif algorithms > 2:
        if experiments < algorithms * 2:
            if rank_option == 'auto':
                rank_option = 'Friedman AR'
            elif rank_option != 'Friedman AR':
                print("Attenzione, con più algoritmi e pochi esperimenti è consigliato utilizzare il test di Friedman con Average Ranks.")
        else:
            if rank_option == 'auto':
                rank_option = 'Friedman'
            elif rank_option != 'Friedman':
                print("Attenzione, con più algoritmi e molti esperimenti è consigliato utilizzare il test di Friedman.")
    
    match rank_option:
        case 'Friedman':
            for csv, df in dataframes.items():
                stat, p_value = friedmanchisquare(*[df[col] for col in df.columns])
                if not p_value < ALPHA_VALUE:
                    raise ValueError(f"Friedman test failed for {csv} with p-value: {p_value}")
                ranks = rankdata(*[df[col] for col in df.columns], method='average') # Rank di Friedman
                
        
        case _:
            raise ValueError("Invalid rank_option provided.")