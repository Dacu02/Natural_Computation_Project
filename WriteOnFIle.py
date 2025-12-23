from copy import deepcopy
from itertools import product
import os

import numpy as np
from Framework import AlgorithmStructure, SEEDS, PROBLEMS
from Algorithms import ParticleSwarmOptimization, ArtificialBeeColony, DifferentialEvolution
import json
def main():
    
    class Combine():
        """
            Classe di comodo per rappresentare combinazioni di parametri di algoritmi
        """
        def __init__(self, list:list):
            """
                Args:
                    list (list): Parametri da combinare
            """
            self._list = list

        def next_element(self):
            for element in self._list:
                yield element

    def expand_algorithms(list_algorithms:list[AlgorithmStructure]) -> list[AlgorithmStructure]:
        """
            Funzione di comodo per espandere tutte le possibili combinazioni specificate tra i parmaetri degli algoritmi
            In particolare, cerca le istanze di Combine e genera tutte le combinazioni possibili.
        """
        expanded = []

        for entry in list_algorithms:
            args = entry["args"]

            # Trova le chiavi che usano Combine
            combine_keys = [k for k, v in args.items() if isinstance(v, Combine)]

            if not combine_keys:
                # Nessuna combinazione → mantieni così com’è
                expanded.append(entry)
                continue

            # Recupera tutte le liste di combinazioni
            combine_lists = [list(args[k].next_element()) for k in combine_keys]

            # Prodotto cartesiano delle combinazioni
            for combo in product(*combine_lists):
                new_entry = deepcopy(entry)
                new_args = new_entry["args"]

                # Sostituisci i Combine con i valori specifici della combinazione
                for k, v in zip(combine_keys, combo):
                    new_args[k] = v

                # Aggiorna il nome rendendolo unico
                new_entry["name"] = new_entry["name"] + "_" + "_".join(
                    f"{k}{v}" for k, v in zip(combine_keys, combo)
                )

                expanded.append(new_entry)

        return expanded

    list_algorithms:list[AlgorithmStructure] = [{
        'algorithm': ParticleSwarmOptimization,
        'args': {
            'pop': Combine(np.arange(20, 101, 20).tolist()),
            'graph': 'Random',
            'lw': 1.3,
            'gw': 1.3,
            'w': 0.8,
            'k': 1,
        },
        'name': 'Random'
        }]



    all_algorithms = expand_algorithms(list_algorithms)
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