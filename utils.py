from copy import deepcopy
from itertools import product
from typing import List, Type, TypedDict
from pyparsing import Any
from Algorithms import Algorithm
import pandas as pd


class AlgorithmStructure(TypedDict):
    algorithm: Type[Algorithm]
    args: dict[str, Any]
    name: str


def load_from_csv(file_path: str, algorithm_class: Type[Algorithm], critical_difference:float, max_algorithms:int|None = None) -> List[AlgorithmStructure]:
    """
        Funzione di comodo per estrarre gli algoritmi da un file CSV dove sono specificati i parametri e gli algoritmi sono ordinti per rank.
        Con la critical_difference si considerano equivalenti tutti gli algoritmi che hanno un rank entro la critical_difference dal primo classificato.
        Di questi algoritmi equivalenti, se ce ne sono più di max_algorithms, si prendono i primi max_algorithms (ordinati per rank).
        
        Args:
            file_path (str): Percorso del file CSV da cui estrarre gli algoritmi
            algorithm_class (Type[Algorithm]): Classe dell'algoritmo da istanziare
            max_algorithms (int): Numero massimo di algoritmi da estrarre (primi max_algorithms per rank)
            critical_difference (float): Differenza di rank entro cui considerare gli algoritmi equivalenti
    """
    df = pd.read_csv(file_path)
    if 'Rank' not in df.columns:
        raise ValueError("Il file CSV deve contenere una colonna 'Rank' per ordinare gli algoritmi.")
    
    # Ordina per Rank
    df_sorted = df.sort_values(by='Rank')
    
    # Se critical_difference è specificata, filtra gli algoritmi equivalenti
    if critical_difference > 0:
        best_rank = df_sorted['Rank'].iloc[0]
        df_sorted = df_sorted[df_sorted['Rank'] <= best_rank + critical_difference]
    
    # Prendi i primi max_algorithms
    if max_algorithms is not None:
        df_top = df_sorted.head(max_algorithms)
    else:
        df_top = df_sorted
    
    algorithms = []
    for i, row in df_top.iterrows():
        args = {col: row[col] for col in df.columns if col != 'Rank' and col != 'thresholds'}
        parsed_args = algorithm_class.parse_args(args)
        algorithms.append({
            'algorithm': algorithm_class,
            'args': parsed_args,
            'name': 'Rank' + str(row['Rank']) + '_' + str(i)
        })
    
    return algorithms

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
            if 'wrapper' in new_args:
                for key, value in new_args['wrapper'].items():
                    new_args[key] = value
                del new_args['wrapper']
                
            # Aggiorna il nome rendendolo unico
            name_parts = []
            for k, v in zip(combine_keys, combo):
                if isinstance(v, dict):
                    name_parts.extend(f"{k2}{v2}" for k2, v2 in v.items())
                else:
                    name_parts.append(f"{k}{v}")
            new_entry["name"] = new_entry["name"] + "_" + "_".join(name_parts)

            expanded.append(new_entry)

    return expanded