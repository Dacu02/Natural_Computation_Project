from re import M
from typing import Literal
from pyswarms.single.general_optimizer import GeneralOptimizerPSO # Graoh specifico
from pyswarms.backend.topology import Pyramid, Star, Ring, VonNeumann, Random

from Algorithms import Algorithm, Problem

SOLUTIONS_BOUNDARY_STRATEGIES = Literal[
    "nearest",      # Reposition the particle to the nearest bound.

    "random",       # Reposition the particle randomly in between the bounds.

    "shrink",       # Shrink the velocity of the particle such that it lands on the bounds.

    "reflective",   # Mirror the particle position from outside the bounds to inside the bounds

    "intermediate",  # Reposition the particle to the midpoint between its current position on the bound surpassing axis and the bound itself. 
                    # This only adjusts the axes that surpass the boundaries.

    "periodic"      # This method resets particles that exceed the bounds to an intermediate position between the boundary and their earlier position. Namely, it changes
                    #the coordinate of the out-of-bounds axis to the middle value between the previous position and the boundary of the axis.
] # Strategie per la gestione delle particelle che superano i limiti, commenti dalla doc.

SPEED_BOUNDARY_STRATEGIES = Literal[
    "unmodified", # Returns the unmodified velocites.
    
    "adjust", # Returns the velocity that is adjusted to be the distance between the current and the previous position.
    
    "invert", # Inverts and shrinks the velocity by the factor -z.
    
    "zero", # Sets the velocity of out-of-bounds particles to zero.
    
] # Strategie per la gestione delle velocità delle particelle che superano i limiti, commenti dalla doc.

HYPER_PARAMETERS_STRATEGIES = Literal[
    "exp_decay", # Decreases the parameter exponentially between limits.

    "lin_variation", # Decreases/increases the parameter linearly between limits.

    "random", # takes a uniform random value between (0.5,1)

    "nonlin_mod", # Decreases/increases the parameter between limits according to a nonlinear modulation index.

] # Strategie per la gestione dei parametri a runtime, commenti dalla doc.

TOPOLOGIES = Literal[
    "Pyramid",
    "Star",
    "Ring",
    "VonNeumann",
    "Random"
] # Tipi di topologie disponibili per il PSO


from pyswarms.backend.topology.random import Random as PSRandom
import numpy as np

# Hard Fix della topologia Random per risolvere errore di compatibilità Numpy 
def patched_compute_gbest(self, swarm, k, **kwargs):
    # chiama la logica originale fino al punto di costruzione di neighbor_idx,
    # oppure ricostruisci neighbor_idx e poi esegui il resto come nell'esempio precedente.
    adj_matrix = self._Random__compute_neighbors(swarm, k)  # se disponibile
    self.neighbor_idx = [adj_matrix[i].nonzero()[0] for i in range(swarm.n_particles)]
    # poi replica il resto del comportamento originale (vedi esempio di subclass)
    idx_min = np.array([swarm.pbest_cost[self.neighbor_idx[i]].argmin() for i in range(len(self.neighbor_idx))])
    best_neighbor = np.array([self.neighbor_idx[i][idx_min[i]] for i in range(len(self.neighbor_idx))]).astype(int)
    best_cost = np.min(swarm.pbest_cost[best_neighbor])
    best_pos = swarm.pbest_pos[best_neighbor]
    return (best_pos, best_cost)

PSRandom.compute_gbest = patched_compute_gbest


class ParticleSwarmOptimization(Algorithm):
    """
        Classe per rappresentare l'algoritmo di Particle Swarm Optimization (PSO).
    """
    def __init__(self, 
                 problem:Problem, 
                 pop:int, 
                 generations:int, 
                 seed:int, 
                 graph:TOPOLOGIES, 
                 lw:float, 
                 gw:float, 
                 w:float, 
                 verbose:bool=False,
                 k:int|None=None,
                 p:int|None=None,
                 r:int|None=None,
                 solut_s:SOLUTIONS_BOUNDARY_STRATEGIES = "shrink",
                 speed_s:SPEED_BOUNDARY_STRATEGIES = 'unmodified',
                 hyper_s:dict[str, HYPER_PARAMETERS_STRATEGIES] = {},
                 vel_clamp:tuple[float, float]|None=None,
                 static:bool=True
                 ):
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
                k (int): Numero di vicini per le topologie che lo richiedono (Ring, VonNeumann, Random).
                p (int): Norma L1 o L2 per le topologie che lo richiedono (Ring, VonNeumann).
                r (int): Raggio per la topologia VonNeumann.
                solution_boundary_strategy (str|None): Strategia per la gestione delle particelle che superano i limiti.
                speed_boundary_strategy (str|None): Strategia per la gestione delle velocità che superano i limiti. Obbligatorio se velocity_clamp è specificato.
                hyper_parameters_strategy (str|None): Strategia per la gestione dei parametri a runtime.
                velocity_clamp (tuple|None): Limiti per la velocità delle particelle.
                static (bool): Se True, le connessioni tra particelle rimangono le stesse durante l'esecuzione.

        """
        super().__init__(problem, pop, generations, seed, verbose)
        self._options = {
            'c1': lw,
            'c2': gw,
            'w': w
        }

        if p and p not in [1, 2]:
            raise ValueError("Parameter 'p' must be either 1 or 2.")

        match graph:
            case "Pyramid":
                self._topology = Pyramid(static=static)
            case "Star":
                self._topology = Star(static=static)
            case "Ring":
                self._topology = Ring(static=static)
                if not k or not p:
                    raise ValueError("For 'Ring' topology, both 'k' and 'p' parameters must be specified and non-zero.")
                self._options['k'] = k  # Numero di vicini per Ring
                self._options['p'] = p  # Norma L1 o L2 per Ring
            case "VonNeumann":
                self._topology = VonNeumann(static=static)
                if not p or not r:
                    raise ValueError("For 'VonNeumann' topology, both 'p' and 'r' parameters must be specified and non-zero.")
                self._options['p'] = p  # Norma L1 o L2 per VonNeumann
                self._options['r'] = r  # Raggio per VonNeumann
            case "Random":
                if not k:
                    raise ValueError("For 'Random' topology, 'k' parameter must be specified and non-zero.")
                self._topology = Random(static=static)
                self._options['k'] = k  # Numero di vicini per Random
            case _:
                raise ValueError("Invalid topology type. Choose from 'Pyramid', 'Star', 'Ring', 'VonNeumann', 'Random'.")
        
        if vel_clamp is not None and speed_s is None:
            raise ValueError("If 'velocity_clamp' is specified, 'speed_boundary_strategy' must also be provided.")

        kwargs = {}
        if vel_clamp is not None:
            kwargs['velocity_clamp'] = vel_clamp
        if speed_s is not None:
            kwargs['vh_strategy'] = speed_s
        if solut_s is not None:
            kwargs['bh_strategy'] = solut_s
        if hyper_s is not None:
            kwargs['oh_strategy'] = hyper_s

        self._pso = GeneralOptimizerPSO(
            n_particles=self._population,
            dimensions=self._dimensions,
            options=self._options,
            bounds=self._bounds,
            topology=self._topology,
            oh_strategy=kwargs,
            velocity_clamp=vel_clamp,
            bh_strategy=solut_s,
            vh_strategy=speed_s,

        )

    def run(self):
        """
            Esegue l'algoritmo PSO.
        """
        return self._pso.optimize(self._function, iters=self._generations, verbose=self._verbose)

    def _set_problem(self, problem:Problem):
        """
            Imposta il problema specifico per l'algoritmo PSO.
            Args:
                problem (Problem): Il problema da risolvere.
        """
        self._bounds = (problem.lb, problem.ub)
        self._dimensions = problem.n_var
        self._function = problem.function