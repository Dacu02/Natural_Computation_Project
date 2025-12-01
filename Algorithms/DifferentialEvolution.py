import numpy as np

from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination

from Algorithms import Algorithm, Problem
from typing import Literal

from pymoo.core.repair import Repair

class Clipping(Repair):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        X = np.clip(X, problem.xl, problem.xu)
        return X

class CustomProblem(PymooProblem):
    def __init__(self, problem:Problem):
        self._function = problem.function
        super().__init__(n_var=problem.n_var, n_obj=1, n_constr=0, xl=problem.lb, xu=problem.ub)
    
    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self._function(X)

def n_ary_tournament(pop, P, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors > pop.size:
        raise Exception("Max pressure greater than pop.size not allowed for tournament!")

    S = np.full(n_tournaments, -1, dtype=int)

    # now do all the tournaments
    for i in range(n_tournaments):
        selected = P[i]
        winner = selected[0]
        for j in range(n_competitors):
            # if the first individiual is better, choose it
            if pop[winner].F > pop[selected[j]].F:
                winner = selected[j]
        S[i] = winner
    return S


class DifferentialEvolution(Algorithm):
    """
        Classe per rappresentare l'algoritmo di Differential Evolution (DE).
    """
    def __init__(self, 
                 problem:Problem, 
                 population:int, 
                 generations:int, 
                 seed:int, 
                 CR:float, 
                 F:float,
                 variant:Literal["DE/rand/1/bin", "DE/best/1/bin"] = "DE/rand/1/bin", 
                 elitism:int=0, 
                 sampling_random:bool=False, 
                 dither:Literal["vector", "scalar"]="vector",
                 jitter:bool=False,
                 verbose:bool=False
                ):
        """
            Inizializza l'algoritmo di Differential Evolution (DE) con i parametri specificati.
            Args:
                problem (Problem): Il problema di ottimizzazione da risolvere.
                population (int): La dimensione della popolazione.
                generations (int): Il numero di generazioni.
                seed (int): Il seme per la generazione di numeri casuali.
                CR (float): La probabilità di crossover.
                F (float): Il differential weight.
                variant (str): La variante di DE da utilizzare.
                elitism (int): Il numero di individui elitari da mantenere.
                sampling_random (bool): Se True, utilizza campionamento casuale invece di Latin Hypercube Sampling.
                dither (str): Tipo di dithering da utilizzare.
                jitter (bool): Se True, abilita il jittering.
                verbose (bool): Se True, abilita l'output dettagliato.
            """

        super().__init__(problem, population, generations, seed, verbose)
        self._de = DE(
            pop_size=self._population, 
            variant=variant, 
            CR=CR, 
            F=F, 
            jitter=jitter, 
            sampling=FloatRandomSampling() if sampling_random else LatinHypercubeSampling(),  # type: ignore
            dither=dither,
            elitism=elitism,
            repair=Clipping(),
        ) 

    def _set_problem(self, problem:Problem):
        self._problem = CustomProblem(problem)

    def run(self):
        return minimize(
            self._problem, 
            self._de, 
            MaximumGenerationTermination(n_max_gen=self._generations),
            seed=self._seed, 
            verbose=True,
        )