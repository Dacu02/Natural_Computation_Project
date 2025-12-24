from copy import deepcopy
import json
import os
from typing import Any, Dict, Type
import fcntl
import numpy as np
from Framework import EarlyStop, load_gnbg_instance, Problem, BOUNDS_MULTIPLIER, GNBG
from Algorithms import Algorithm, ParticleSwarmOptimization, ArtificialBeeColony, DifferentialEvolution
import multiprocessing as mp

EXP_PATH = os.path.join("experiments", "todo")
RESULT_PATH = os.path.join("experiments", "results")

def execution(seed:int, problem:int, alg_class:Type[Algorithm], alg_args:Dict[str, Any], file:str):
    np.random.seed(seed)
    instance = load_gnbg_instance(problemIndex=problem)
    lb = -BOUNDS_MULTIPLIER*np.ones(instance.Dimension)
    ub = BOUNDS_MULTIPLIER*np.ones(instance.Dimension)
    alg_args['generations'] = instance.MaxEvals // alg_args['pop']
    problem_custom_instance = Problem(function=instance.fitness, n_var=instance.Dimension, lb=lb, ub=ub)
    print(f"\nEsecuzione dell'algoritmo {alg_class.__name__} sul problema f{problem} con seed {seed}")
    alg_args['seed'] = seed
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

    with open(os.path.join(file), 'w') as f:
            f.write('LastFunctionEvaluation,MinimumError\n')
            f.write(f"{instance.FE},{convergence[-1]}\n")
def loop():
    while True:
        if not os.path.exists(EXP_PATH):
            print("Esperimenti completati.")
            return
        for problem_directory in os.listdir(EXP_PATH):
            if not os.path.isdir(os.path.join(EXP_PATH, problem_directory)):    
                continue

            problem_path = os.path.join(EXP_PATH, problem_directory)
            if not os.path.isdir(problem_path):
                continue

            for algorithm_directory in os.listdir(problem_path):
                algorithm_path = os.path.join(problem_path, algorithm_directory)
                if not os.path.isdir(algorithm_path):
                    continue

                for seed_directory in os.listdir(algorithm_path):
                    seed_path = os.path.join(algorithm_path, seed_directory)
                    if not os.path.isdir(seed_path):
                        continue

                    config_file = os.path.join(seed_path, 'config.json')
                    if not os.path.isfile(config_file):
                        config_file = os.path.join(seed_path, 'config.json.lock')
                        if not os.path.isfile(config_file):
                            continue

                    os.makedirs(os.path.join(RESULT_PATH, problem_directory, algorithm_directory, seed_directory), exist_ok=True)
                    result_file = os.path.join(RESULT_PATH, problem_directory, algorithm_directory, f'{str(seed_directory)}.csv')

                    try:
                        #os.rename(config_file, config_file + ".lock") # lock
                        with open(config_file, 'r') as f:
                            fcntl.flock(f, fcntl.LOCK_EX)
                            config = json.load(f)
                            seed = config['seed']
                            problem = int(config['problem'])
                            alg_name = config['class']
                            alg_args = config['parameters']

                            alg_class = Algorithm.read_name(alg_name)
                            print(f"Avviando esecuzione per problema {problem}, algoritmo {alg_name}, seed {seed}...")
                            try:
                                execution(seed, problem, alg_class, deepcopy(alg_args), result_file)
                            except Exception as e:
                                print(f"Errore durante l'esecuzione: {e}")
                                print(f"Si salta l'algoritmo {alg_name} per il problema {problem} con seed {seed}.")
                        fcntl.flock(f, fcntl.LOCK_UN)
                        os.remove(config_file)
                        #os.removedirs(seed_path)
                    except:
                        continue


def main():
    PROCESS_COUNT = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count())) # HPC Unisa, altrimenti locale
    print(f"Using {PROCESS_COUNT} processes for parallel execution.")
    procs = []
    for _ in range(PROCESS_COUNT): 
        p = mp.Process(target=loop) 
        p.start() 
        procs.append(p) 
    try: 
        for p in procs: 
            p.join() 
    except KeyboardInterrupt: 
        print("Interrotto dall'utente, terminazione in corso...") 
        for p in procs: 
            p.terminate()

if __name__ == "__main__":
    main()

