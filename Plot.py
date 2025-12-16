import os
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Plot di confronto delle convergence summary tra algoritmi
def summary_plots(problem_folder: str, algorithms: list[str], results_per_algorithm: dict[str, np.ndarray], problem: int, output_folder:str|None=None) -> None:
    """
        Genera immagini e csv per mostrare il risultato di esperimenti tra più algoritmi
        Args:
    """
    seeds_count = results_per_algorithm[algorithms[0]].shape[0]

    if not output_folder:
        output_folder = problem_folder
    else:
        os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        alg_folder = os.path.join(output_folder, algorithm)
        summary_df = read_csv(os.path.join(alg_folder, 'results_summary.csv'))
        generations = summary_df['Generation'].values
        mean_errors = summary_df['MeanError'].values
        mean_se = summary_df['MeanError'].values
        ci95_high = mean_errors + (mean_se) * stats.t.ppf(0.975, df=seeds_count - 1) # type: ignore
        ci95_low = mean_errors - (mean_se) * stats.t.ppf(0.975, df=seeds_count - 1) # type: ignore
        plt.plot(generations, mean_errors, linewidth=2, label=algorithm) # type: ignore
        plt.fill_between(generations, ci95_low, ci95_high, alpha=0.2) # type: ignore
    plt.xlabel('Generations')
    plt.ylabel('Mean Error')
    plt.title(f'Convergence Summary Comparison on f{problem}')
    plt.savefig(os.path.join(output_folder, f'convergence_summary_comparison.png'))

    # Generazione dei summary tra differenti algoritmi di ciascun problema
    with open(os.path.join(output_folder, f'final_results_summary.csv'), 'w') as f:
        f.write('Algorithm,MeanFinalError,StdFinalError,MeanSE,ShapiroWilk,FinalResults\n')
        for alg_name, results_array in results_per_algorithm.items():
            if results_array is not None:
                mean_final_error = np.mean(results_array)
                std_final_error = np.std(results_array, ddof=1)
                sem_final_error = std_final_error / np.sqrt(len(results_array))
                results_str = ';'.join(map(str, results_array))
                _, shapiro_p = stats.shapiro(results_array)
                f.write(f"{alg_name},{mean_final_error},{std_final_error},{sem_final_error},{shapiro_p},{results_str}\n")
                if shapiro_p < 0.05:
                    print(f"Warning: i risultati finali dell'algoritmo {alg_name} sul problema f{problem} sembrano non seguire una distribuzione normale (Shapiro-Wilk p={shapiro_p:.4f})")
    
    # Matrice di confronto t-test tra algoritmi
    n_algs = len(algorithms)
    comparison_matrix = np.zeros((n_algs, n_algs))

    for i_idx, alg_i in enumerate(algorithms):
        for j_idx, alg_j in enumerate(algorithms):
            if i_idx != j_idx:
                results_i = results_per_algorithm[alg_i]
                results_j = results_per_algorithm[alg_j]
                if results_i is not None and results_j is not None:
                    stat, p_value_two_tailed = stats.ttest_ind(results_i, results_j)
                    p_value_one_tailed = p_value_two_tailed / 2 # type: ignore
                    if np.mean(results_i) < np.mean(results_j):
                        p_value_one_tailed = 1 - p_value_one_tailed
                    comparison_matrix[i_idx, j_idx] = p_value_one_tailed
            else:
                comparison_matrix[i_idx, j_idx] = 0.5  # diagonale

    # Salva matrice come immagine
    plt.figure(figsize=(8, 6))
    plt.imshow(comparison_matrix, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(comparison_matrix):
        plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', weight=1000)
    plt.colorbar(label='p-value')
    plt.xticks(range(n_algs), algorithms, ha='right')
    plt.yticks(range(n_algs), algorithms)
    plt.title('T-test p-values between algorithms ($H_0: \\mu_{row} > \\mu_{col}$)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'algorithms_comparison_matrix.png'), dpi=150)
    plt.clf()

    # Plot delle distribuzioni finali degli errori come normali, per ogni algoritmo
    for alg_name, results_array in results_per_algorithm.items():
        if results_array is not None:
            mean = np.mean(results_array)
            std = np.std(results_array, ddof=1)
            mean_se = std / np.sqrt(len(results_array))
            x = np.linspace(mean - 4*std, mean + 4*std, 100)
            y = stats.norm.pdf(x, mean, std)
            
            plt.figure(figsize=(10, 6))
            # Normal distribution curve
            plt.plot(x, y, 'r-', linewidth=2, label=f'$\\mathcal{{N}}({mean:.2f}, {std:.2f})$')
            # Plot individual results
            #plt.plot(results_array, stats.norm.pdf(results_array, mean, std), 'x', alpha=1, markersize=9, label='Results')
            # Plot histogram
            plt.hist(results_array, bins='auto', density=True, alpha=0.5, label='Histogram of Results')
            plt.title(f'Final Errors Distribution for {alg_name} on f{problem}')
            plt.xlabel('Final Error')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, alg_name, f'final_error_distribution_{alg_name}.png'))
            plt.clf()

    # Plot delle distribuzioni finali degli errori come normali, per tutti gli algoritmi
    plt.figure(figsize=(10, 6))
    for i, (alg_name, results_array) in enumerate(results_per_algorithm.items()):
        if results_array is not None:
            mean = np.mean(results_array)
            std = np.std(results_array, ddof=1)
            mean_se = std / np.sqrt(len(results_array))
            x = np.linspace(mean - 4*std, mean + 4*std, 100)
            y = stats.norm.pdf(x, mean, std)
            plt.plot(x, y, label=f'{alg_name} $\\mathcal{{N}}({mean:.2f}, {std:.2f})$') 
    plt.title(f'Final Errors Distribution of all algorithms on f{problem}')
    plt.xlabel('Final Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'final_error_distribution_all_algorithms.png'))
    plt.clf()

    # Plot sovrapposto delle convergence summary tra algoritmi
    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(algorithms):
        alg_folder = os.path.join(output_folder, algorithm)
        summary_df = read_csv(os.path.join(alg_folder, 'results_summary.csv'))
        generations = summary_df['Generation'].values
        mean_errors = summary_df['MeanError'].values
        mean_se = summary_df['MeanError'].values
        #ci95_high = mean_errors + (mean_se) * stats.t.ppf(0.975, df=seeds_count - 1)  # type: ignore
        #ci95_low = mean_errors - (mean_se) * stats.t.ppf(0.975, df=seeds_count - 1) # type: ignore
        plt.plot(generations, mean_errors, linewidth=2, label=algorithm) # type: ignore
        #plt.fill_between(generations, ci95_low, ci95_high, alpha=0.2) # type: ignore
        plt.xlabel('Generations')
        plt.ylabel('Mean Error')
    plt.title(f'Convergence Summary Comparison on f{problem}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'convergence_summary_comparison.png'))
    plt.clf()


def main():
    data_path = input("Inserisci il percorso nel quale generare i plot: ") # ./experiment_folder/problem_folder
    problem_index = int(os.path.basename(data_path).replace('f_', ''))
    if not os.path.exists(data_path):
        raise ValueError(f"Il percorso {data_path} non esiste.")
    sub_folders = [f.path for f in os.scandir(data_path) if f.is_dir()]
    
    plot_name = input("Inserisci il nome da assegnare alla cartella dei plot: ")
    while plot_name is None or plot_name in [sub_folders]:
        plot_name = input("Inserisci un nome differente: ")
    plot_folder = os.path.join(data_path, plot_name)
    os.makedirs(plot_folder, exist_ok=True)

    chosen_algorithms = []
    while True:
        alg_name = input("Inserisci il nome di un algoritmo da includere, oppure premi invio per terminare: ")
        if alg_name == "":
            break

        if alg_name not in chosen_algorithms and alg_name in [os.path.basename(f) for f in sub_folders]:
            chosen_algorithms.append(alg_name) 
        else:
            print("Algoritmo non riconosciuto o già inserito.")

    if len(chosen_algorithms) == 0:
        raise ValueError("Devi inserire almeno un algoritmo da includere nei plot.")

    final_results_per_algorithm: dict[str, np.ndarray|None] = {algorithm: None for algorithm in chosen_algorithms}
    for algorithm in chosen_algorithms:
        alg_folder = os.path.join(data_path, algorithm)
        result_files = [f.path for f in os.scandir(alg_folder) if f.is_file() and f.name.endswith('.csv')]
        dfs = []
        for file in result_files:
            dfs.append(read_csv(file))

        length = len(dfs[0])
        for df in dfs:
            if len(df) != length:
                raise ValueError("All result CSV files must have the same number of rows.")
            
        arr = np.stack([df.iloc[:, 1].values for df in dfs])  # shape (n_runs, n_gens)
        final_results_per_algorithm[algorithm] = (arr[:, -1])  # prendi l'ultimo valore di ogni run
    print("Costruzione dei plot in corso...")
    summary_plots(data_path, chosen_algorithms, final_results_per_algorithm, problem=problem_index, output_folder=plot_folder) # type: ignore

if __name__ == "__main__":
    main()