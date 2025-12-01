from abc import ABC

from Algorithms import Problem

class Algorithm(ABC):
    """
        Classe astratta per rappresentare un algoritmo generico.
    """
    def __init__(self, problem:Problem, population:int, generations:int, seed:int, verbose:bool=False):
        """
            Inizializza l'algoritmo con i parametri di base.
            Args:
                problem (Problem): Il problema da risolvere.
                population (int): La dimensione della popolazione.
                generations (int): Il numero di generazioni.
                seed (int): Il seme per la generazione casuale.
                verbose (bool): Se True, abilita l'output dettagliato.
            """
        self._set_problem(problem)
        self._population = population
        self._generations = generations
        self._seed = seed
        self._verbose = verbose

    def run(self):
        raise NotImplementedError("Abstract method 'run' must be implemented in subclasses.")

    def _set_problem(self, problem:Problem):
        raise NotImplementedError("Abstract method '_set_problem' must be implemented in subclasses.")