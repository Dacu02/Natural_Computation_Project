from copy import deepcopy
from itertools import product
import os

import numpy as np
from Framework import AlgorithmStructure, SEEDS, PROBLEMS
from Algorithms import ParticleSwarmOptimization, ArtificialBeeColony, DifferentialEvolution
import json

from experiments import EXPERIMENTS
def main():
    
    all_algorithms = EXPERIMENTS
    os.path.join(os.getcwd(), 'experiments')
    for problem in PROBLEMS:
        for algorithm in all_algorithms:
            for seed in SEEDS:
                dir_path = os.path.join('experiments', 'todo', str(problem), algorithm['name'], str(seed))
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, 'config.json')
                with open(file_path, 'w') as f:
                    config = {
                        'seed': seed,
                        'problem': str(problem),
                        'algorithm': algorithm['name'],
                        'class': algorithm['algorithm'].get_name(),
                        'parameters': algorithm['args']
                    }
                    json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()