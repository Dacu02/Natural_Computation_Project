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
    
    @classmethod
    def get_name(cls) -> str:
        """
            Restituisce il nome dell'algoritmo.
            Returns:
                str: Il nome dell'algoritmo.
        """
        match cls.__name__:
            case "ParticleSwarmOptimization":
                return "PSO"
            case "ArtificialBeeColony": 
                return "ABC"
            case "DifferentialEvolution":
                return "DE"
            case _:
                raise ValueError(f"Algoritmo '{cls.__name__}' non riconosciuto.")

    @staticmethod
    def read_name(name:str) -> type['Algorithm']:
        """
            Restituisce la classe dell'algoritmo.
            Returns:
                type[Algorithm]: La classe dell'algoritmo.
        """
        from Algorithms import ParticleSwarmOptimization, ArtificialBeeColony, DifferentialEvolution
        
        match name:
            case "PSO":
                return ParticleSwarmOptimization
            case "ABC": 
                return ArtificialBeeColony
            case "DE":
                return DifferentialEvolution
            case _:
                raise ValueError(f"Algoritmo '{name}' non riconosciuto.")
        

    @staticmethod
    def parse_args(args:dict) -> dict:
        """
            Funzione di comodo per parsare gli argomenti da un dizionario, convertendo i tipi se necessario.
            Args:
                args (dict): Il dizionario degli argomenti da parsare.
            Returns:
                dict: Il dizionario degli argomenti parsati.
        """
        raise NotImplementedError("Abstract method 'parse_args' must be implemented in subclasses.")