from utils import AlgorithmStructure, Combine, expand_algorithms, load_from_csv
from Algorithms import Algorithm, ParticleSwarmOptimization, ArtificialBeeColony, DifferentialEvolution
LB, UB = -100, 100
RANGE = UB - LB

################## DEFINE THE EXPERIMENTS #########################


_list_algorithms: dict[int, list[AlgorithmStructure]] = {
}
####################################################################



EXPERIMENTS = {}
for key in _list_algorithms:
    EXPERIMENTS[key] = expand_algorithms(_list_algorithms[key])