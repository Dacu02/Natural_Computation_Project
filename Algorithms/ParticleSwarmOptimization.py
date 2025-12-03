from typing import Literal
from pyswarms.single.general_optimizer import GeneralOptimizerPSO # Graoh specifico
from pyswarms.backend.topology import Pyramid, Star, Ring, VonNeumann, Random

from Algorithms import Algorithm, Problem

SOLUTIONS_BOUNDARY_STRATEGIES = Literal[
    "Nearest", # Reposition the particle to the nearest bound.
    "Random", # Reposition the particle randomly in between the bounds.
    "Shrink", # Shrink the velocity of the particle such that it lands on the bounds.
    "Reflective", # Mirror the particle position from outside the bounds to inside the bounds
    "Intermediate" # Reposition the particle to the midpoint between its current position on the bound surpassing axis and the bound itself. This only adjusts the axes that surpass the boundaries.
] # Strategie per la gestione delle particelle che superano i limiti, commenti dalla doc.

SPEED_BOUNDARY_STRATEGIES = Literal[
    "Unmodified", # Returns the unmodified velocites.
    "Adjust", # Returns the velocity that is adjusted to be the distance between the current and the previous position.
    "Invert", # Inverts and shrinks the velocity by the factor -z.
    "Zero", # Sets the velocity of out-of-bounds particles to zero.
] # Strategie per la gestione delle velocità delle particelle che superano i limiti, commenti dalla doc.

HYPER_PARAMETERS_STRATEGIES = Literal[
    "exp_decay", # Decreases the parameter exponentially between limits.
    "lin_variation", # Decreases/increases the parameter linearly between limits.
    "random", # takes a uniform random value between (0.5,1)
    "nonlin_mod", # Decreases/increases the parameter between limits according to a nonlinear modulation index.
] # Strategie per la gestione dei parametri a runtime, commenti dalla doc.
class ParticleSwarmOptimization(Algorithm):
    """
        Classe per rappresentare l'algoritmo di Particle Swarm Optimization (PSO).
    """
    def __init__(self, 
                 problem:Problem, 
                 population:int, 
                 generations:int, 
                 seed:int, 
                 topology:Literal["Pyramid", "Star", "Ring", "VonNeumann", "Random"], 
                 local_weight:float, 
                 global_weight:float, 
                 inertia:float, 
                 verbose:bool=False,
                 k:int=0,
                 p:int=0,
                 r:int=0):
        """
            Inizializza l'algoritmo del PSO con i parametri specificati.
            Args:
                problem (Problem): Il problema da risolvere.
                population (int): La dimensione della popolazione (numero di particelle).
                generations (int): Il numero di generazioni (iterazioni).
                seed (int): Il seme per la generazione di numeri casuali.
                topology (str): La topologia da utilizzare per il PSO.
                local_weight (float): Attrazione verso l'ottimo personale
                global_weight (float): Attrazione verso l'ottimo locale/globale
                inertia (float): Il peso di inerzia.
                verbose (bool): Se True, abilita l'output dettagliato.

        """
        super().__init__(problem, population, generations, seed, verbose)
        self._options = {
            'c1': local_weight,
            'c2': global_weight,
            'w': inertia
        }

        if topology == "Pyramid":
            self._topology = Pyramid()
        elif topology == "Star":
            self._topology = Star()
        elif topology == "Ring":
            self._topology = Ring()
            self._options['k'] = k  # Numero di vicini per Ring
            self._options['p'] = p  # Norma L1 o L2 per Ring
        elif topology == "VonNeumann":
            self._topology = VonNeumann()
            self._options['k'] = k  # Numero di vicini per VonNeumann
            self._options['p'] = p  # Norma L1 o L2 per VonNeumann
            self._options['r'] = r  # Raggio per VonNeumann
        elif topology == "Random":
            self._topology = Random()
            self._options['k'] = k  # Numero di vicini per Random
        else:
            raise ValueError("Invalid topology type. Choose from 'Pyramid', 'Star', 'Ring', 'VonNeumann', 'Random'.")
       
        self._pso = GeneralOptimizerPSO(
            n_particles=self._population,
            dimensions=self._dimensions,
            options=self._options,
            bounds=self._bounds,
            topology=self._topology
        )

    def run(self):
        """
            Esegue l'algoritmo PSO.
        """
        
        return self._pso.optimize(self._function, iters=self._generations, verbose=self._verbose, n_processes=4)

    def _set_problem(self, problem:Problem):
        """
            Imposta il problema specifico per l'algoritmo PSO.
            Args:
                problem (Problem): Il problema da risolvere.
        """
        self._bounds = (problem.lb, problem.ub)
        self._dimensions = problem.n_var
        self._function = problem.function