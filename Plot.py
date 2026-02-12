import os
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

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

    # Grafico di confronto delle convergence summary tra algoritmi
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        alg_folder = os.path.join(problem_folder, algorithm)
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
    plt.close()

    # Generazione dei summary tra differenti algoritmi di ciascun problema
    with open(os.path.join(output_folder, f'final_results_summary.csv'), 'w') as f:
        f.write('Algorithm,MeanFinalError,StdFinalError,MeanSE,MedianSE,ShapiroWilk,FinalResults\n')
        for alg_name, results_array in results_per_algorithm.items():
            if results_array is not None:
                mean_final_error = np.mean(results_array)
                std_final_error = np.std(results_array, ddof=1)
                sem_final_error = std_final_error / np.sqrt(len(results_array))
                median_se = np.median(results_array)
                results_str = ';'.join(map(str, results_array))
                _, shapiro_p = stats.shapiro(results_array)
                f.write(f"{alg_name},{mean_final_error},{std_final_error},{sem_final_error},{median_se},{shapiro_p},{results_str}\n")
                if shapiro_p < 0.05:
                    print(f"Warning: i risultati finali dell'algoritmo {alg_name} sul problema f{problem} sembrano non seguire una distribuzione normale (Shapiro-Wilk p={shapiro_p:.4f})")

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
            image_output_folder = os.path.join(output_folder, alg_name)
            os.makedirs(image_output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, alg_name, f'final_error_distribution_{alg_name}.png'))
            plt.close()

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
    plt.close()

    # Plot sovrapposto delle convergence summary tra algoritmi
    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(algorithms):
        alg_folder = os.path.join(problem_folder, algorithm)
        summary_df = read_csv(os.path.join(alg_folder, 'results_summary.csv'))
        generations = summary_df['Generation'].values
        mean_errors = summary_df['MeanError'].values
        mean_se = summary_df['MeanError'].values
        ci95_high = mean_errors + (mean_se) * stats.t.ppf(0.975, df=seeds_count - 1)  # type: ignore
        ci95_low = mean_errors - (mean_se) * stats.t.ppf(0.975, df=seeds_count - 1) # type: ignore
        plt.plot(generations, mean_errors, linewidth=2, label=algorithm) # type: ignore
        plt.fill_between(generations, ci95_low, ci95_high, alpha=0.2) # type: ignore
        plt.xlabel('Generations')
        plt.ylabel('Mean Error')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
    plt.title(f'Convergence Summary Comparison on f{problem}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'convergence_summary_comparison.png'))
    plt.close()


def main():
    args = sys.argv
    problem_indexes = []
    data_path = input("Inserisci la cartella dei risultati dal quale estrapolare i dati: ") if len(args) < 2 else args[1]
    for sub_folder in [f.path for f in os.scandir(data_path)]:
        try:
            problem_index = int(os.path.basename(sub_folder).replace('f_', ''))
            problem_indexes.append(problem_index)
        except:
            pass

    if not problem_indexes or len(problem_indexes) == 0:
        raise ValueError('Errore nel caricamento dei dati')

    #plot_name = input("Inserisci il nome da assegnare alla cartella dei plot: ") if len(args) < 3 else args[2]
    plot_folder = os.path.join(data_path)#, plot_name)
    os.makedirs(plot_folder, exist_ok=True)
    chosen_algorithms = []
    if len(args) == 3:
        if args[2] == '.':
            chosen_algorithms = [sf.name for sf in os.scandir(os.path.join(data_path, 'f_'+str(problem_indexes[0])))]
        else:
            chosen_algorithms = [args[2]]
    elif len(args) > 2:
        chosen_algorithms = args[2:]
    else:
        while True:
            alg_name = input("Inserisci il nome di un algoritmo da includere, oppure premi invio per terminare: ")
            
            if alg_name == "":
                break

            elif alg_name not in chosen_algorithms:# and alg_name in [os.path.basename(f) for f in [sub_]]:
                chosen_algorithms.append(alg_name) 
            else:
                print("Algoritmo non riconosciuto o già inserito.")

        if len(chosen_algorithms) == 0:
            raise ValueError("Devi inserire almeno un algoritmo da includere nei plot.")

    print("Costruzione dei plot in corso...")
    for problem_index in problem_indexes:
        final_results_per_algorithm: dict[str, np.ndarray|None] = {algorithm: None for algorithm in chosen_algorithms}
        for algorithm in chosen_algorithms:
            problem_data_path = os.path.join(data_path, 'f_'+str(problem_index))
            alg_folder = os.path.join(problem_data_path, algorithm)
            print(alg_folder)
            result_files = [f.path for f in os.scandir(alg_folder) if f.is_file() and f.name.endswith('.csv') and not 'summary' in f.name]
            dfs = []
            for file in result_files:
                dfs.append(read_csv(file))
            length = len(dfs[0])
            for df in dfs:
                if len(df) != length:
                    raise ValueError("All result CSV files must have the same number of rows.")
                
            arr = np.stack([df.iloc[:, 1].values for df in dfs])  # shape (n_runs, n_gens)
            final_results_per_algorithm[algorithm] = (arr[:, -1])  # prendi l'ultimo valore di ogni run

        summary_plots(problem_data_path, chosen_algorithms, final_results_per_algorithm, problem=problem_index, output_folder=os.path.join(plot_folder, 'f_'+str(problem_index))) # type: ignore

if __name__ == "__main__":
    main()