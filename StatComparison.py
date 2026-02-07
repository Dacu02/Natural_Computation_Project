import os
import pandas as pd
import numpy as np
from pingouin import friedman
from scipy.stats import rankdata, chi2, studentized_range
from matplotlib import pyplot as plt
from typing import Literal
TEST_OPTIONS = Literal['Friedman', 'Wilcoxon', 'auto']
AGGLOMERATION_METHOD = Literal['mean', 'median']
ALPHA_VALUE = 0.05


def CompareAlgorithms(
    data_path:str, 
    test_option:TEST_OPTIONS='auto', 
    agglomeration_method:AGGLOMERATION_METHOD='mean',
    result_threshold:float=1e-8,
    minimum_threshold_count:int=6,
):
        
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
        dataframes[csv_name] = df

    algorithms, cols = dataframes[next(iter(dataframes))].shape
    experiments = len([seed for seed in dataframes[next(iter(dataframes))].columns if seed.isdigit()])
    param_cols = [col for col in dataframes[next(iter(dataframes))].columns if not col.isdigit()]
    for df in dataframes.values():
        if df.shape != (algorithms, cols):
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
    else:
        if test_option == 'auto':
            test_option = 'Friedman'
        elif test_option != 'Friedman':
            print("Attenzione, con più algoritmi è consigliato utilizzare il test di Friedman.")
    
    match test_option:
        case 'Friedman':
            for csv, df in dataframes.items():
                
                seed_cols = [col for col in df.columns if str(col).strip().isdigit()]
                df['Algorithm_ID'] = df[param_cols].astype(str).agg('_'.join, axis=1)
                
                df_long = df.melt(
                    id_vars=['Algorithm_ID'], 
                    value_vars=seed_cols, 
                    var_name='Seed_Subject', 
                    value_name='Score'
                )
                
                # dv: La variabile dipendente (Score)
                # within: Il fattore che stiamo confrontando (gli Algoritmi, cioè le righe)
                # subject: Il fattore di blocco (i Semi/Seeds, cioè le colonne originali)
                res = friedman(data=df_long, dv='Score', within='Algorithm_ID', subject='Seed_Subject', method='f')
                
                stat = res['F'].values[0]
                p_value = res['p-unc'].values[0]

                print(f"Friedman test for {csv}: statistic={stat}, p-value={p_value}")
                
                if not float(p_value) < ALPHA_VALUE: 
                    raise ValueError(f"Friedman test failed for {csv} with p-value: {p_value}")
                
                df_ranking = df.set_index('Algorithm_ID')[seed_cols]                
                ranks_matrix = df_ranking.rank(axis=0, method='average', ascending=True) 
                mean_ranks = ranks_matrix.mean(axis=1).sort_values()  

                mean_ranks_df = mean_ranks.to_frame(name='Rank')
                mean_ranks_df['thresholds'] = mean_ranks_df.index.map(
                    lambda alg_id: (df_ranking.loc[alg_id] < result_threshold).sum()
                )
                
                split_params = mean_ranks_df.index.to_series().str.split('_', expand=True)
                for i, arg in enumerate(param_cols):
                    # Usiamo .iloc[:, i] per prendere la colonna i-esima in modo sicuro
                    if i < split_params.shape[1]:
                        mean_ranks_df[arg] = split_params.iloc[:, i]

                mean_ranks_df = mean_ranks_df.drop(columns=['index'], errors='ignore')
                mean_ranks_df = mean_ranks_df.reset_index(drop=True)
                mean_ranks_df = mean_ranks_df.sort_values(by='Rank')

                
                filtered_ranks = mean_ranks_df#[mean_ranks_df['thresholds'] >= minimum_threshold_count]
                print(f"Mean ranks for {csv} (filtered):")
                os.makedirs(os.path.join(data_path, 'data_analysis'), exist_ok=True)
                with open(os.path.join(data_path, 'data_analysis', f"{csv}_mean_ranks.csv"), 'w') as f:
                    filtered_ranks.to_csv(f, index=False)

                DOF = np.inf
                qa = studentized_range.ppf(1 - ALPHA_VALUE, algorithms, DOF)
                critical_difference = qa * np.sqrt(algorithms * (algorithms + 1) / (6 * experiments))
                print(f"Critical Difference for {csv}: {critical_difference}")
                
                # Convertiamo population in numerico per avere l'asse Y ordinato correttamente
                if 'population' in filtered_ranks.columns:
                    filtered_ranks['population'] = pd.to_numeric(filtered_ranks['population'], errors='coerce')
                else:
                    raise ValueError(f"'population' column is missing in the data for {csv}. Please ensure it is included in the CSV files.")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Mappatura colori richiesta
                topo_colors = {
                    'Torus': 'red', 
                    'Random': 'green', 
                    'Star': 'blue'
                }
                
                # Verifica esistenza colonne necessarie
                has_topology = 'topology' in filtered_ranks.columns
                has_population = 'population' in filtered_ranks.columns

                if not has_topology and has_population:
                    ax.scatter(
                        x=filtered_ranks['Rank'],
                        y=filtered_ranks['population'],
                        c='gray',
                        s=100,
                        alpha=0.8
                    )
                    ax.set_ylabel('population')
                    ax.set_xlabel('Mean Rank')
                    ax.set_title(f'Algorithm Ranking (Lower is better) - {csv}')
                    ax.grid(True, linestyle='--', alpha=0.5)

                elif has_topology and has_population:
                    # Raggruppa per topologia per assegnare colori e label
                    for topology_name, group in filtered_ranks.groupby('topology'):
                        color = topo_colors.get(topology_name, 'gray') # grigio se topologia non riconosciuta
                        
                        ax.scatter(
                            x=group['Rank'],
                            y=group['population'],
                            c=color,
                            label=topology_name,
                            s=100,
                            alpha=0.8
                        )
                    
                    ax.set_ylabel('population')
                    ax.legend(title='topology', loc='upper right', bbox_to_anchor=(1, 0.85))
                    ax.set_xlabel('Mean Rank')
                    ax.set_title(f'Algorithm Ranking (Lower is better) - {csv}')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    
                    # Invertiamo asse X se si vuole il rango 1 a sinistra (solitamente preferito)
                    # Se Rank 1 è il migliore, i valori bassi devono stare a sinistra/basso.
                    # Qui lasciamo standard: sinistra = rank basso (migliore), destra = rank alto (peggiore).
                
                else:
                    raise ValueError(f"Required column 'population' is missing in the data for {csv}.")
                    
                # --- INIZIO BLOCCO CRITICAL DIFFERENCE ---
                # Calcoliamo i limiti attuali del grafico per posizionare la barra
                # Otteniamo i limiti attuali dopo aver plottato tutti i punti
                y_limits = ax.get_ylim()
                y_range = y_limits[1] - y_limits[0]

                # POSIZIONE: 
                cd_x_start = 0
                cd_x_end = cd_x_start + critical_difference
                cd_y_pos = y_limits[0] + (y_range * 0.9)

                # 1. Disegna la linea orizzontale (il segmento CD)
                ax.plot([cd_x_start, cd_x_end], [cd_y_pos, cd_y_pos], 
                        color='black', linewidth=1.5)

                # 2. Disegna le stanghette verticali alle estremità (per sembrare un righello)
                tick_height = y_range * 0.05 # Altezza stanghetta proporzionale all'asse Y
                ax.plot([cd_x_start, cd_x_start], [cd_y_pos - tick_height, cd_y_pos + tick_height], 
                        color='black', linewidth=1.5)
                ax.plot([cd_x_end, cd_x_end], [cd_y_pos - tick_height, cd_y_pos + tick_height], 
                        color='black', linewidth=1.5)

                # 3. Aggiungi il testo sopra la barra
                ax.text((cd_x_start + cd_x_end) / 2, cd_y_pos - tick_height, 
                        f'CD = {critical_difference:.2f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
                # --- FINE BLOCCO CRITICAL DIFFERENCE ---

                plt.tight_layout()
                plt.savefig(os.path.join(data_path, 'data_analysis', f"{csv}_critical_difference.png"), dpi=300)
                plt.close()


        case _:
            raise ValueError("Invalid test_option provided.")
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        data_path = sys.argv[1]
        test_option = sys.argv[2]
    else:
        raise ValueError("Please provide the data path and test option as command line arguments.")
    
    CompareAlgorithms(data_path, test_option=test_option) # type: ignore

# 0.05 0.03 0.02