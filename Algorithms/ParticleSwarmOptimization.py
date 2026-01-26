import math
from typing import Literal
import pyswarms.backend
from pyswarms.single.general_optimizer import GeneralOptimizerPSO
from pyswarms.backend.topology import Pyramid, Star, Ring, VonNeumann, Random, Topology
from Algorithms import Algorithm, Problem
import pyswarms

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
    "Random",
    "Torus"
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

class Torus(Topology):
    """
        Classe per rappresentare una topologia toroidale.
    """
    def __init__(self, size:int):
        """
            Inizializza la topologia toroidale. Questa topologia viene definia manualmente siccome pyswarms non include una reale topologia di Von Neumann, in particolare toroidale.
            In una rete toroidale ciascun nodo ha esattamente quattro vicini:
            1. Il nodo a sinistra (se esiste, altrimenti l'ultimo nodo della riga)
            2. Il nodo a destra (se esiste, altrimenti il primo nodo della riga)
            3. Il nodo sopra (se esiste, altrimenti l'ultimo nodo della colonna)
            4. Il nodo sotto (se esiste, altrimenti il primo nodo della colonna)
            Args:
                static (bool): Se True, le connessioni tra particelle rimangono le stesse durante l'esecuzione.
        """
        super().__init__(static=False)
        if size < 9:
            raise ValueError("Size must be at least 9 to form a torus topology.")
        n, m = self.best_rectangle(size)
        grid_indices = np.arange(size).reshape((n, m))
        # Roll effettua il wrap-around tipico 
        idx_up = np.roll(grid_indices, 1, axis=0).flatten()
        idx_down = np.roll(grid_indices, -1, axis=0).flatten()
        idx_left = np.roll(grid_indices, 1, axis=1).flatten()
        idx_right = np.roll(grid_indices, -1, axis=1).flatten()
        idx_self = grid_indices.flatten()
        self.neighbor_idx = np.column_stack((idx_self, idx_up, idx_down, idx_left, idx_right))

    def compute_gbest(self, swarm, **kwargs):
        all_costs = swarm.pbest_cost[self.neighbor_idx]
        local_best = np.argmin(all_costs, axis=1) # La riga è il vicinato
        best_neighbor_indices = self.neighbor_idx[np.arange(len(self.neighbor_idx)), local_best] 
        best_pos = swarm.pbest_pos[best_neighbor_indices]
        return best_pos, np.min(swarm.pbest_cost[best_neighbor_indices])


    def compute_position(self, swarm, bounds=None, bh=pyswarms.backend.handlers.BoundaryHandler(strategy="periodic")):
        return pyswarms.backend.operators.compute_position(swarm, bounds, bh)
    
    def compute_velocity(self, swarm, clamp=None, vh=pyswarms.backend.handlers.VelocityHandler(strategy="unmodified"), bounds=None):
        return pyswarms.backend.operators.compute_velocity(swarm, clamp, vh, bounds)
    
    @staticmethod
    def best_rectangle(n: int) -> tuple[int, int]: # type: ignore
        """
            Metodo statico per calcolare le dimensioni del rettangolo più vicino al quadrato per un dato numero di particelle.
            Args:
                n (int): Numero di particelle.
            Returns:
                tuple: Una tupla contenente le dimensioni (larghezza, altezza) del rettangolo.
            Raises:
                ValueError: Se n non è un intero positivo.
        """
        if n <= 0:
            raise ValueError("n deve essere un intero positivo")

        root = int(math.isqrt(n))
        for a in range(root, 0, -1):
            if n % a == 0:
                b = n // a
                return a, b 


class ParticleSwarmOptimization(Algorithm):
    """
        Classe per rappresentare l'algoritmo di Particle Swarm Optimization (PSO).
    """
    def __init__(
            self, 
            problem:Problem, 
            population:int, 
            generations:int, 
            seed:int, 
            topology:TOPOLOGIES, 
            local_weight:float, 
            global_weight:float, 
            inertia:float, 
            verbose:bool=False,
            k:int|None=5,
            p:int|None=None,
            r:int|None=None,
            solution_boundary_strategy:SOLUTIONS_BOUNDARY_STRATEGIES = "shrink",
            speed_strategy:SPEED_BOUNDARY_STRATEGIES = 'unmodified',
            end_inertia:float|None=None,
            end_local_weight:float|None=None,
            end_global_weight:float|None=None,
            velocity_clamp:tuple[float, float]|None=None,
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
                end_inertia (float|None): Valore finale per l'inerzia, se si vuole variare nel tempo.
                end_local_weight (float|None): Valore finale per il peso locale, se si vuole variare nel tempo.
                end_global_weight (float|None): Valore finale per il peso globale, se si vuole variare nel tempo.
                velocity_clamp (tuple|None): Limiti per la velocità delle particelle.
                static (bool): Se True, le connessioni tra particelle rimangono le stesse durante l'esecuzione.

        """
        super().__init__(problem, population, generations, seed, verbose)
        self._options = {
            'c1': local_weight,
            'c2': global_weight,
            'w': inertia
        }

        if p and p not in [1, 2]:
            raise ValueError("Parameter 'p' must be either 1 or 2.")

        match topology:
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
            case "Torus":
                self._topology = Torus(size=self._population)
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
        
        if velocity_clamp is not None and speed_strategy is None:
            raise ValueError("If 'velocity_clamp' is specified, 'speed_boundary_strategy' must also be provided.")

        kwargs = {}
        kwargs['oh_strategy'] = {}

        if end_inertia is not None:
            kwargs['oh_strategy']['w'] = ('lin_variation', (inertia, end_inertia))
        if end_local_weight is not None:
            kwargs['oh_strategy']['c1'] = ('lin_variation', (local_weight, end_local_weight))
        if end_global_weight is not None:
            kwargs['oh_strategy']['c2'] = ('lin_variation', (global_weight, end_global_weight))

        self._pso = GeneralOptimizerPSO(
            n_particles=self._population,
            dimensions=self._dimensions,
            options=self._options,
            bounds=self._bounds,
            topology=self._topology,
            oh_strategy=kwargs,
            velocity_clamp=velocity_clamp,
            bh_strategy=solution_boundary_strategy,
            vh_strategy=speed_strategy,

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