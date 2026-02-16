from copy import deepcopy
from itertools import product
from typing import List, Type, TypedDict

from pyparsing import Any
from Algorithms import Algorithm, ParticleSwarmOptimization, ArtificialBeeColony, DifferentialEvolution
import pandas as pd
LB, UB = -100, 100
RANGE = UB - LB


class AlgorithmStructure(TypedDict):
    algorithm: Type[Algorithm]
    args: dict[str, Any]
    name: str

_list_algorithms: dict[int, List[AlgorithmStructure]] = {
-1: [{
    'algorithm': ParticleSwarmOptimization,
    'args': {
        'population': [11*11, 13*13],
        'topology': ['Random', 'Star', 'Torus'],
        'local_weight': [.75, 1.25],
        'global_weight': [.75, 1.25],

        'inertia': [.7, .9],
        'velocity_clamp': (-.15 * RANGE, .15 * RANGE),
        'end_inertia': [.4, None],
    },
    'name': 'PSO',
}]}
EXPERIMENTS = _list_algorithms