from typing import Literal
import numpy as np
import copy
from beeoptimal.abc import ArtificialBeeColony as _ABC
from beeoptimal.bee import Bee
from Algorithms import Problem, Algorithm

def get_iabc_donor(abc: _ABC, bee_idx: int , population: list) -> Bee: 
    fit = np.array([bee.fitness for bee in population])
    ts = np.quantile(fit, 0.25) #0.25 

    memory = [b for i, b in enumerate(population) 
              if fit[i] < ts and i != bee_idx]
    
    if len(memory) == 0:
        memory = [b for i, b in enumerate(population) if i != bee_idx]
    return copy.deepcopy(np.random.choice(memory))
def patch_iabc(abc: _ABC):
   bee_opt_methos = abc.get_candidate_neighbor_

   def ibac_get_candidate_neighbor(self, bee, bee_idx, population):
       if self._mutation_strategy == 'iABC':
           donor = get_iabc_donor(self, bee_idx, population)
           candidate = copy.deepcopy(bee)

           j = np.random.randint(0, self.dim)
           phi = np.random.uniform(-self.sf, self.sf)
           candidate.position[j] = [
               bee.position[j] + 
               phi * (bee.position[j] - donor.position[j])
           ]

           candidate.position = np.clip(candidate.position, self.bounds[:, 0], self.bounds[:, 1])
           return candidate
       if self._mutation_strategy == 'iABC-block':
           donor = get_iabc_donor(self, bee_idx, population)
           candidate = copy.deepcopy(bee)
           block_size = max(2, int(0.1 * self.dim))  # Dimensione del blocco (10% delle dimensioni, minimo 2)
           idx = np.random.choice(self.dim, block_size, replace=False)  # Indici del blocco da modificare
           phi = np.random.uniform(-self.sf, self.sf, size=block_size)

           candidate.position[idx] = [
               bee.position[idx] +
               phi * (bee.position[idx] - donor.position[idx])
           ]
           candidate.position = np.clip(candidate.position, self.bounds[:, 0], self.bounds[:, 1])
           return candidate
       return bee_opt_methos(bee, bee_idx, population)
   abc.get_candidate_neighbor_ = ibac_get_candidate_neighbor        



class ArtificialBeeColony(Algorithm):
    """Classe per l'algoritmo Artificial Bee Colony."""
    def __init__(
            self, 
            problem: Problem, 
            population: int, 
            generations: int, 
            seed: int, 
            max_scouts: int, 
            verbose: bool = False, 
            limit: int|None = None,
            selection_strategy: Literal['RouletteWheel', 'Tournament'] = 'RouletteWheel',
            mutation_strategy: Literal[
                'StandardABC', 
                'ModifiedABC', 
                'ABC/best/1', 
                'ABC/best/2'
                ] = 'StandardABC',
            initialization_strategy: Literal['random', 'chaotic'] = 'random',
            tournament_size: int|None = None,
        ):
        """
        Inizializza l'algoritmo ABC con i parametri specificati.
        Args:
            problem (Problem): Il problema da risolvere.
            population (int): La dimensione della popolazione (numero di api).
            generations (int): Il numero di generazioni (iterazioni).
            seed (int): Il seme per la generazione di numeri casuali.
            max_scouts (int): Numero massimo di api esploratrici.
            verbose (bool): Se True, abilita l'output dettagliato.
            limit (int|None): Limite per l'abbandono di una soluzione.
            selection_strategy (str): Strategia di selezione ('RouletteWheel' o 'Tournament')
            mutation_strategy (str): Strategia di mutazione ('StandardABC', 'ModifiedABC', 'ABC/best/1', 'ABC/best/2').
            initialization_strategy (str): Strategia di inizializzazione ('random' o 'chaotic')
            tournament_size (int|None): Dimensione del torneo se si utilizza la selezione a torneo.
        """
        super().__init__(problem, population, generations, seed, verbose)
        self.max_scouts = max_scouts
        lower_bounds = problem.lb
        upper_bounds = problem.ub
        self._bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(problem.n_var)]
        self._limit = limit
        self._selection_strategy = selection_strategy
        if tournament_size is not None and selection_strategy != 'Tournament':
            raise ValueError("La dimensione del torneo è specificata ma la strategia di selezione non è 'Tournament'.")
        self._mutation_strategy = mutation_strategy
        self._initialization_strategy = initialization_strategy
        self._tournament_size = tournament_size
        self._abc = _ABC(
            colony_size=self._population,
            function=lambda x: problem.function(x).item(),
            bounds=np.array(self._bounds),
            max_scouts=self.max_scouts

        )
        patch_iabc(self._abc)
    
    def _set_problem(self, problem: Problem):
        pass

    def run(self):
        self._abc.optimize(
            max_iters=int(self._generations),
            limit=self._limit if self._limit else 'default',
            selection=self._selection_strategy,
            mutation=self._mutation_strategy,
            initialization=self._initialization_strategy,
            tournament_size=self._tournament_size if self._selection_strategy == 'Tournament' else None,
            verbose=self._verbose,
            random_seed=self._seed
        ) 
        return self._abc.optimal_bee.position, self._abc.optimal_bee.value
    

@staticmethod
def parse_args(args: dict) -> dict:
    """
    Funzione di comodo per parsare i parametri da un dizionario (ad esempio estratto da un file CSV).
    Args:
        args (dict): Dizionario contenente i parametri da parsare.
    Returns:
        dict: Dizionario con i parametri parsati e convertiti nei tipi corretti.
    """
    parsed_args = {
        'population': int(args['population']),
        'generations': int(args['generations']),
        'seed': int(args['seed']),
        'max_scouts': int(args['max_scouts']),
        'verbose': bool(int(args['verbose'])),
        'limit': int(args['limit']), 
        'selection_strategy': args['selection_strategy'],
        'mutation_strategy': args['mutation_strategy'],
        'initialization_strategy': args['initialization_strategy'],
        'tournament_size': int(args['tournament_size']) if 'tournament_size' in args and args['tournament_size'] != 'nan' else None,
    }
    return parsed_args


        
        
         




