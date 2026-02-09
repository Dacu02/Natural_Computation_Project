from copy import deepcopy
from multiprocessing.pool import AsyncResult
import os
from time import strftime
from Plot import summary_plots
import numpy as np
from scipy import stats
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')   # compatibilità multiprocessing (non grafica su schermo)
import matplotlib.pyplot as plt
import multiprocessing as mp
from pandas import read_csv
from Algorithms import Algorithm, Problem, DifferentialEvolution, ParticleSwarmOptimization, ArtificialBeeColony
from typing import Any, Dict, List, Type, Type, TypedDict
from experiments import AlgorithmStructure, EXPERIMENTS
import sys
import traceback
class EarlyStop(Exception):
    """Eccezione per arrestare l'esecuzione anticipatamente."""
    pass

SEEDS: list[int] = [5751, 94862, 48425, 79431, 28465, 917654, 468742131, 745612, 1354987, 126879]
PROBLEMS: list[int] = [18,19,20]
BOUNDS_MULTIPLIER = 100
FUNCTIONS_PATH = os.path.join(os.getcwd())
# Define the GNBG class
class GNBG:
    def __init__(self, MaxEvals, AcceptanceThreshold, Dimension, CompNum,
                 MinCoordinate, MaxCoordinate, CompMinPos, CompSigma,
                 CompH, Mu, Omega, Lambda, RotationMatrix,
                 OptimumValue, OptimumPosition):
        self.MaxEvals = MaxEvals
        self.AcceptanceThreshold = AcceptanceThreshold
        self.Dimension = Dimension
        self.CompNum = CompNum
        self.MinCoordinate = MinCoordinate
        self.MaxCoordinate = MaxCoordinate
        self.CompMinPos = CompMinPos
        self.CompSigma = CompSigma
        self.CompH = CompH
        self.Mu = Mu
        self.Omega = Omega
        self.Lambda = Lambda
        self.RotationMatrix = RotationMatrix
        self.OptimumValue = OptimumValue
        self.OptimumPosition = OptimumPosition
        self.FEhistory = []
        self.FE = 0
        self.AcceptanceReachPoint = np.inf
        self.BestFoundResult = np.inf
        self.BestFoundPosition = None  # traccia della best position

    def fitness(self, X):
        if len(X.shape) < 2:
            X = X.reshape(1, -1)
        SolutionNumber = X.shape[0]
        result = np.nan * np.ones(SolutionNumber)

        for jj in range(SolutionNumber):
            self.FE += 1
            if self.FE > self.MaxEvals:
                if np.isinf(self.BestFoundResult):
                    self.BestFoundResult = 1e30
                raise EarlyStop("Raggiunto MaxEvals")

            x = X[jj, :].reshape(-1, 1)
            f = np.nan * np.ones(self.CompNum)

            for k in range(self.CompNum):
                if len(self.RotationMatrix.shape) == 3:
                    rotation_matrix = self.RotationMatrix[:, :, k]
                else:
                    rotation_matrix = self.RotationMatrix

                a = self.transform(
                    (x - self.CompMinPos[k, :].reshape(-1, 1)).T @ rotation_matrix.T,
                    self.Mu[k, :],
                    self.Omega[k, :],
                )
                b = self.transform(
                    rotation_matrix @ (x - self.CompMinPos[k, :].reshape(-1, 1)),
                    self.Mu[k, :],
                    self.Omega[k, :],
                )

                sigma_k = self.CompSigma[k].item()
                lambda_k = self.Lambda[k].item()
                quad = (a @ np.diag(self.CompH[k, :]) @ b).item()
                f[k] = sigma_k + (quad ** lambda_k)

            result[jj] = np.min(f)

            # storia
            self.FEhistory = np.append(self.FEhistory, result[jj])

            # aggiorna best valore e posizione
            if self.BestFoundResult > result[jj]:
                self.BestFoundResult = result[jj]
                # salva la soluzione corrente in forma 1D
                self.BestFoundPosition = x.flatten().copy()

            # controllo AcceptanceThreshold sul best
            error = abs(self.BestFoundResult - self.OptimumValue)
            if error < self.AcceptanceThreshold and np.isinf(self.AcceptanceReachPoint):
                self.AcceptanceReachPoint = self.FE
                #raise EarlyStop("Raggiunta AcceptanceThreshold") 
                # Sufficienti rirsorse per continuare l'esecuzione

        return result

    def transform(self, X, Alpha, Beta):
        Y = X.copy()
        tmp = (X > 0)
        Y[tmp] = np.log(X[tmp])
        Y[tmp] = np.exp(Y[tmp] + Alpha[0] * (np.sin(Beta[0] * Y[tmp]) + np.sin(Beta[1] * Y[tmp])))
        tmp = (X < 0)
        Y[tmp] = np.log(-X[tmp])
        Y[tmp] = -np.exp(Y[tmp] + Alpha[1] * (np.sin(Beta[2] * Y[tmp]) + np.sin(Beta[3] * Y[tmp])))
        return Y

def load_gnbg_instance(problemIndex: int) -> GNBG:
        if 1 <= problemIndex <= 24:
            filename = os.path.join('functions', f'f{problemIndex}.mat')
            GNBG_tmp = loadmat(os.path.join(FUNCTIONS_PATH, filename))['GNBG']
            MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
            AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
            Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
            CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
            MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
            MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
            CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
            CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
            CompH = np.array(GNBG_tmp['Component_H'][0, 0])
            Mu = np.array(GNBG_tmp['Mu'][0, 0])
            Omega = np.array(GNBG_tmp['Omega'][0, 0])
            Lambda = np.array(GNBG_tmp['lambda'][0, 0])
            RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
            OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
            OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
        else:
            raise ValueError('ProblemIndex must be between 1 and 24.')

        #print(f"Loaded GNBG problem instance f{problemIndex} with dimension {Dimension} and {CompNum} components.\n"
         #   f"Global optimum value: {OptimumValue} AcceptanceThreshold: {AcceptanceThreshold}, MaxEvals: {MaxEvals}")

        return GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)


if __name__ == '__main__':
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    
    PROCESS_COUNT = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count())) # HPC Unisa, altrimenti locale
    print(f"Using {PROCESS_COUNT} processes for parallel execution.")
    
    MINIMAL_FRAMEWORK = True  # Se true, disabilita stampe extra e dati non essenziali

    
    def execution(seed:int, problem:int, alg_class:Type[Algorithm], alg_args:Dict[str, Any], description:str, folder:str):
        np.random.seed(seed)  
        instance = load_gnbg_instance(problemIndex=problem)
        lb = -BOUNDS_MULTIPLIER*np.ones(instance.Dimension)
        ub = BOUNDS_MULTIPLIER*np.ones(instance.Dimension)
        alg_args['generations'] = instance.MaxEvals // alg_args['population']
        problem_custom_instance = Problem(function=instance.fitness, n_var=instance.Dimension, lb=lb, ub=ub)
        print(f"\nEsecuzione dell'algoritmo {alg_class.__name__} sul problema f{problem} con seed {seed}")
        alg_instance = alg_class(problem_custom_instance, **alg_args)
        
        try:
            alg_instance.run()

        except EarlyStop as e:
            print(f"Algoritmo fermato anticipatamente")
        
        convergence = []
        best_error = float('inf')
        for value in instance.FEhistory:
            error = abs(value - instance.OptimumValue)
            if error < best_error:
                best_error = error
            convergence.append(best_error)


        #print(f"Best found objective value: {instance.BestFoundResult} at position {instance.BestFoundPosition}")
        #print(f"Error from the global optimum: {abs(instance.BestFoundResult - instance.OptimumValue)}")
        #print(f"Function evaluations to reach acceptance threshold: {instance.AcceptanceReachPoint if not np.isinf(instance.AcceptanceReachPoint) else 'Not reached'}")
        #print("FE usate:", instance.FE)
        #print("MaxEvals:", instance.MaxEvals)

        if not MINIMAL_FRAMEWORK:
            # Plotting the convergence
            plt.plot(range(1, len(convergence) + 1), convergence)
            plt.xlabel('Function Evaluation Number (FE)')
            plt.ylabel('Error')
            plt.title('Convergence Plot')
            plt.yscale('log')  # Set y-axis to logarithmic scale
            plt.savefig(os.path.join(folder, f'convergence_{description}.png'))
            plt.clf()  # Clear the figure for the next plot
            with open(os.path.join(folder, f'results_{description}.csv'), 'w') as f:
                f.write('FunctionEvaluation,Error\n')
                for fe, err in enumerate(convergence):
                    f.write(f"{fe},{err}\n")
        else:
            with open(os.path.join(folder, f'results_{description}.csv'), 'w') as f:
                f.write('LastFunctionEvaluation,MinimumError,Seed\n')
                f.write(f"{instance.FE},{convergence[-1]},{alg_args['seed']}\n")

    if len(sys.argv) == 4:
        min_index, max_index, file_name = sys.argv[1:4]
        problem_class = (PROBLEMS[0] - 1) // 8 + 1
        different_classes = False
        for problem in PROBLEMS:
            if problem_class != (problem - 1) // 8 + 1:
                different_classes = True

        if different_classes:
            algorithms: List[AlgorithmStructure] = EXPERIMENTS[-1]
        else:
            algorithms: List[AlgorithmStructure] = EXPERIMENTS[retrieve_class(PROBLEMS[0])]
    else:
        print("Usage: python Framework.py <start_index> <end_index> <file_name>")
        sys.exit(1)
        
    if not min_index.isdigit() or not max_index.isdigit():
        raise ValueError("Gli argomenti devono essere numeri interi.")
    
    min_index = int(min_index)
    max_index = int(max_index)
    if min_index < 0 or max_index < min_index:
        raise ValueError("Intervallo di indici non valido.")
    algorithms = [alg for i, alg in enumerate(algorithms) if i % max_index == min_index]
    run_name = f'partial_run_{min_index}_of_{max_index}__' + file_name

    results = os.path.join(os.getcwd(), 'results', run_name)
    os.makedirs(results, exist_ok=True)
        
    
    print(f"{len(algorithms)} algorithms to execute, each for {len(SEEDS)} seeds for {len(PROBLEMS)} problems.")
    processes: list[AsyncResult] = []
    with mp.Pool(processes=PROCESS_COUNT) as pool:
        for i, problem in enumerate(PROBLEMS):
            problem_folder = os.path.join(results, f'f_{problem}')
            os.makedirs(problem_folder, exist_ok=True)
            for algorithm in algorithms:
                alg_class = algorithm['algorithm']
                alg_args = algorithm['args'].copy()
                description = algorithm.get('name', '_'.join(map(str, alg_args.values())))
                if not description:
                    description = 'Algorithm_' + str(i+1)
                alg_folder = os.path.join(problem_folder, description)
                os.makedirs(alg_folder, exist_ok=True)
                with open(os.path.join(alg_folder, 'config.txt'), 'w') as f:
                    f.write(f"Algorithm: {algorithm['algorithm'].__name__}\n")
                    f.write(f"Arguments: {algorithm['args']}\n")
                for seed in SEEDS:
                    alg_args['seed'] = seed
                    processes.append(pool.apply_async(execution, args=(seed, problem, alg_class, alg_args.copy(), str(seed), alg_folder)))

        # Attende il completamento di tutti i processi
        count = 0
        for p in processes:
            try:
                p.get()
            except Exception as e:
                print(f"Error in a subprocess: {e}")
                with open(os.path.join(os.getcwd(), 'errors.log'), 'a') as f:
                    f.write(f"ERROR: {str(e)}\n")
                    f.write(traceback.format_exc())
            print(f"Completed {count + 1} out of {len(processes)} processes.", flush=True)
            count += 1

    # Dopo la fine di tutte le esecuzioni, genera i summary
    if not MINIMAL_FRAMEWORK:
        for i, problem in enumerate(PROBLEMS):
            problem_folder = os.path.join(results, f'f_{problem}')
            final_results_per_algorithm: dict[str, np.ndarray|None] = {algorithm['name']: None for algorithm in algorithms}
            
            algorithm_unrun = []

            # Gnerazione dei summary tra differenti run di uno stesso algoritmo
            for algorithm in algorithms:
                description = algorithm.get('name', '_'.join(map(str, algorithm['args'].values())))
                if not description:
                    description = 'Algorithm_' + str(i+1)
                alg_folder = os.path.join(problem_folder, description)
                result_files = [f.path for f in os.scandir(alg_folder) if f.is_file() and f.name.endswith('.csv')]
                dfs = []
                for file in result_files:
                    dfs.append(read_csv(file))
                try:
                    length = len(dfs[0])
                    for df in dfs:
                        if len(df) != length:
                            raise ValueError("All result CSV files must have the same number of rows.")
                except IndexError:
                    algorithm_unrun.append(algorithm)
                    continue
                        
                arr = np.stack([df.iloc[:, 1].values for df in dfs])  # shape (n_runs, n_gens)
                final_results_per_algorithm[algorithm['name']] = (arr[:, -1])  # prendi l'ultimo valore di ogni run
                n_runs, n_gens = arr.shape

                mean_errors = arr.mean(axis=0)
                std_errors  = arr.std(axis=0, ddof=1)            # std tra run
                sem = std_errors / np.sqrt(n_runs)               # std della media
                t = stats.t.ppf(0.975, df=n_runs - 1)            # 95% CI
                ci95_low  = mean_errors - t * sem
                ci95_high = mean_errors + t * sem

                x = np.arange(1, n_gens + 1)

                with open(os.path.join(alg_folder, f'results_summary.csv'), 'w') as f:
                    f.write('Generation,MeanError,StdError,MeanEstimatorSE,NRuns\n')
                    for gen in range(n_gens):
                        f.write(f"{gen+1},{mean_errors[gen]},{std_errors[gen]},{sem[gen]},{n_runs}\n")


                plt.figure(figsize=(10, 5))

                # singole run 
                for run in arr:
                    plt.plot(x, run, color='lightblue', alpha=0.25, linewidth=0.8)

                # CI al 95%
                plt.fill_between(x, ci95_low, ci95_high, color='#a6c8ff', alpha=0.5, label='95% CI')

                # Deviazione standard
                plt.fill_between(x, mean_errors - std_errors, mean_errors + std_errors, color='pink', alpha=0.35, label='Standard Deviation')
                
                # Media
                plt.plot(x, mean_errors, color='tab:blue', linewidth=2, label='Mean Error')

                # opzionale: marker con errorbar che mostra SD come barre verticali
                # plt.errorbar(x, mean_errors, yerr=std_errors, fmt='o', color='tab:blue',
                            # ecolor='lightcoral', elinewidth=1, capsize=3, markersize=3, alpha=0.9)

                plt.xlabel('Generations')
                plt.ylabel('Error')
                plt.title(f'Convergence Summary of {description} on f{problem}')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(alg_folder, f'convergence_summary.png'))
                plt.clf()

            for algorithm in algorithm_unrun:
                final_results_per_algorithm.pop(algorithm['name'])
                algorithms.pop(algorithms.index(algorithm))
                print(f"Algorithm {algorithm['name']} was not run on problem f{problem} and will be excluded from the comparison plot.")
    else:
        for _, problem in enumerate(PROBLEMS):
            algorithm_unrun = []
            problem_folder = os.path.join(results, f'f_{problem}')
            alg_errors = {}
            
            for algorithm in algorithms:
                alg_folder = os.path.join(problem_folder, algorithm['name'])
                result_files = [f.path for f in os.scandir(alg_folder) if f.is_file() and f.name.endswith('.csv')]
                alg_errors[algorithm['name']] = {}

                for file in result_files:
                    df = read_csv(file)
                    os.remove(file)
                    result_seed = int(df.iloc[0]['Seed'])
                    result_min_error = float(df.iloc[0]['MinimumError'])
                    alg_errors[algorithm['name']][result_seed] = result_min_error

                os.remove(os.path.join(alg_folder, 'config.txt'))
                os.rmdir(alg_folder)
                if not alg_errors[algorithm['name']]:
                    algorithm_unrun.append(algorithm)

            for algorithm in algorithm_unrun:
                print(f"Algorithm {algorithm['name']} was not run on problem f{problem} and will be excluded from the comparison plot.")
                alg_errors.pop(algorithm['name'])
                algorithms.pop(algorithms.index(algorithm))

                # Rimuovi anche le cartelle degli altri problemi
                for pr in PROBLEMS:
                    pr_folder = os.path.join(results, f'f_{pr}')
                    pr_alg_folder = os.path.join(pr_folder, algorithm['name'])
                    if os.path.exists(pr_alg_folder):
                        for file in os.listdir(pr_alg_folder):
                            os.remove(os.path.join(pr_alg_folder, file))
                        os.rmdir(pr_alg_folder)

            with open(os.path.join(results, f'{problem}_summary_errors.csv'), 'w') as f:
                f.write(f'{",".join(map(str, SEEDS))},{",".join(sorted(algorithms[0]["args"].keys()))}\n')
                for algorithm in algorithms:
                    f.write(f"{','.join(f'{alg_errors[algorithm['name']][seed]}' for seed in SEEDS)},{",".join(str(algorithm['args'][key]).replace(',', ';').replace(' ','') for key in sorted(algorithms[0]['args'].keys()))}\n")
            os.rmdir(os.path.join(results, f'f_{problem}'))