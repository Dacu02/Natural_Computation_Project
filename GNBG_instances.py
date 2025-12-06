import os
from time import strftime
import numpy as np
from scipy import stats
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')   # compatibilità multiprocessing (non grafica su schermo)
import matplotlib.pyplot as plt
import multiprocessing as mp
from pandas import DataFrame, read_csv
from Algorithms import Algorithm, Problem, DifferentialEvolution, ParticleSwarmOptimization, ArtificialBeeColony
from typing import Any, Dict, List, Type, Type, TypedDict
class EarlyStop(Exception):
    """Eccezione per arrestare l'esecuzione anticipatamente."""
    pass

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
                raise EarlyStop("Raggiunta AcceptanceThreshold")

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

class AlgorithmStructure(TypedDict):
    algorithm: Type[Algorithm]
    args: Dict[str, Any]
    name: str|None

if __name__ == '__main__':
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the folder where you want to read/write files
    folder_path = os.path.join(current_dir)

    SEEDS: list[int] = [1,2,3]
    PROBLEMS: list[int] = [1,2]
    PROCESS_COUNT = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count())) # HPC Unisa, altrimenti locale
    print(f"Using {PROCESS_COUNT} processes for parallel execution.")
    BOUNDS_MULTIPLIER = 100
    results = os.path.join(os.getcwd(), 'results', strftime('%d__%H_%M'))
    os.makedirs(results, exist_ok=True)

    def load_gnbg_instance(problemIndex: int):
        if 1 <= problemIndex <= 24:
            filename = os.path.join('functions', f'f{problemIndex}.mat')
            GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
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

        print(f"Loaded GNBG problem instance f{problemIndex} with dimension {Dimension} and {CompNum} components.\n"
            f"Global optimum value: {OptimumValue} AcceptanceThreshold: {AcceptanceThreshold}, MaxEvals: {MaxEvals}")

        return GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)

    
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


        print(f"Best found objective value: {instance.BestFoundResult} at position {instance.BestFoundPosition}")
        print(f"Error from the global optimum: {abs(instance.BestFoundResult - instance.OptimumValue)}")
        print(f"Function evaluations to reach acceptance threshold: {instance.AcceptanceReachPoint if not np.isinf(instance.AcceptanceReachPoint) else 'Not reached'}")
        print("FE usate:", instance.FE)
        print("MaxEvals:", instance.MaxEvals)
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

    algorithms: List[AlgorithmStructure] = [{
        'algorithm': ParticleSwarmOptimization,
        'args': {
            'population': 100,
            'topology': 'VonNeumann',
            'local_weight': 1.5,
            'global_weight': 1.5,
            'inertia': 0.7,
            'p': 1,
            'r': 1,
        },
        'name': 'ParticleSwarmOptimization_VonNeumann'
    }, {
        'algorithm': ParticleSwarmOptimization,
        'args': {
            'population': 100,
            'topology': 'Star',
            'local_weight': 1.5,
            'global_weight': 1.5,
            'inertia': 0.7,
            'k': 3,
            'p': 2,
        },
        'name': 'ParticleSwarmOptimization_Star'
    }]


    for i, problem in enumerate(PROBLEMS):
        problem_folder = os.path.join(results, f'f_{problem}')
        os.makedirs(problem_folder, exist_ok=True)
        for algorithm in algorithms:
            alg_class = algorithm['algorithm']
            alg_args = algorithm['args']
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
                
                with mp.Pool(processes=PROCESS_COUNT) as pool:
                    pool.apply(execution, args=(seed, problem, alg_class, alg_args.copy(), str(seed), alg_folder))
            result_files = [f.path for f in os.scandir(alg_folder) if f.is_file() and f.name.endswith('.csv')]
            dfs = []
            for file in result_files:
                dfs.append(read_csv(file))

            length = len(dfs[0])
            for df in dfs:
                if len(df) != length:
                    raise ValueError("All result CSV files must have the same number of rows.")
                
            arr = np.stack([df.iloc[:, 1].values for df in dfs])  # shape (n_runs, n_gens)
            
            n_runs, n_gens = arr.shape

            mean_errors = arr.mean(axis=0)
            std_errors  = arr.std(axis=0, ddof=1)            # std tra run
            sem = std_errors / np.sqrt(n_runs)               # std della media
            t = stats.t.ppf(0.975, df=n_runs - 1)            # 95% CI
            ci95_low  = mean_errors - t * sem
            ci95_high = mean_errors + t * sem

            x = np.arange(1, n_gens + 1)

            plt.figure(figsize=(10, 5))

            # singole run 
            for run in arr:
                plt.plot(x, run, color='lightblue', alpha=0.25, linewidth=0.8)


            # banda CI 95% (precisione della media)
            plt.fill_between(x, ci95_low, ci95_high,
                            color='tab:pink', alpha=0.25, label='95% CI mean')

            # banda SD (dispersione tra run)
            plt.fill_between(x, mean_errors - std_errors, mean_errors + std_errors,
                            color='tab:gray', alpha=0.4, label='±1 SD (spreading)')
            
            # media
            plt.plot(x, mean_errors, color='tab:blue', linewidth=2, label='Mean Error')

            # opzionale: marker con errorbar che mostra SD come barre verticali
            # plt.errorbar(x, mean_errors, yerr=std_errors, fmt='o', color='tab:blue',
                        # ecolor='lightcoral', elinewidth=1, capsize=3, markersize=3, alpha=0.9)

            plt.xlabel('Generations')
            plt.ylabel('Error')
            plt.title(alg_folder)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(alg_folder, f'convergence_summary.png'))
            plt.clf()
            

# If you use your own algorithm (not from a library), you can use result = gnbg.fitness(X) to calculate the fitness values of multiple solutions stored in a matrix X.
# The function returns the fitness values of the solutions in the same order as they are stored in the matrix X.

# After running the algorithm, the best fitness value is stored in gnbg.BestFoundResult.
# The best found position is stored in gnbg.BestFoundPosition.
# The function evaluation number where the algorithm reached the acceptance threshold is stored in gnbg.AcceptanceReachPoint. If the algorithm did not reach the acceptance threshold, it is set to infinity.
# For visualizing the convergence behavior, the history of the objective values is stored in gnbg.FEhistory, however it needs to be processed as follows:
