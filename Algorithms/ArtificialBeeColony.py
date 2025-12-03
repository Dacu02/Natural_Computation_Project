from typing import Literal
import numpy as np
from beeoptimal.abc import ArtificialBeeColony as _ABC
from beeoptimal.benchmarks import BenchmarkFunction
from Algorithms import Problem, Algorithm
class CustomFunction(BenchmarkFunction):
    def __init__(self, function:callable, bounds):
        super().__init__(fun=function, bounds=np.array(bounds), name="Custom GNBG Function")

    def evaluate(self, x):
        return self.fun(x)
    
class ArtificialBeeColony(Algorithm):
    """Classe per l'algoritmo Artificial Bee Colony."""
    def __init__(
            self, 
            problem: Problem, 
            population: int, 
            generations: int, 
            seed: int, 
            max_scouts:int, 
            verbose: bool = False, 
            limit: int|None = None,
            selection_strategy: Literal['RouletteWheel', 'Tournament'] = 'RouletteWheel',
            mutation_strategy: Literal['StandardABC', 'ModifiedABC', 'ABC/best/1', 'ABC/best/2'] = 'StandardABC',
            initialization_strategy: Literal['random', 'cahotic'] = 'random',
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
        self._mutation_strategy = mutation_strategy
        self._initialization_strategy = initialization_strategy
        self._tournament_size = tournament_size
        self._abc = _ABC(
            colony_size=self._population,
            function=lambda x: problem.function(x).item(),
            bounds=np.array(self._bounds),
            max_scouts=self.max_scouts

        )
    
    def _set_problem(self, problem: Problem):
        #self._custom_function = CustomFunction(fun=problem.function, bounds=self._bounds)
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