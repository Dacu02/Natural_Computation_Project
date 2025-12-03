import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Artificial Bee Colony 
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from Algorithms import DifferentialEvolution, ParticleSwarmOptimization, Problem, ArtificialBeeColony

class EarlyStop(Exception):
    """Eccezione per fermare anticipatamente differential_evolution."""
    pass

# Define the GNBG class
class GNBG:
    def __init__(self, MaxEvals, AcceptanceThreshold, Dimension, CompNum,
                 MinCoordinate, MaxCoordinate, CompMinPos, CompSigma,
                 CompH, Mu, Omega, Lambda, RotationMatrix,
                 OptimumValue, OptimumPosition):
        self.MaxEvals = MaxEvals
        self.AcceptanceThreshold = AcceptanceThreshold
        self.Dimension = Dimension
        self.CompNum = CompNum
        self.MinCoordinate = MinCoordinate
        self.MaxCoordinate = MaxCoordinate
        self.CompMinPos = CompMinPos
        self.CompSigma = CompSigma
        self.CompH = CompH
        self.Mu = Mu
        self.Omega = Omega
        self.Lambda = Lambda
        self.RotationMatrix = RotationMatrix
        self.OptimumValue = OptimumValue
        self.OptimumPosition = OptimumPosition
        self.FEhistory = []
        self.FE = 0
        self.AcceptanceReachPoint = np.inf
        self.BestFoundResult = np.inf
        self.BestFoundPosition = None  # traccia della best position

    def fitness(self, X):
        if len(X.shape) < 2:
            X = X.reshape(1, -1)
        SolutionNumber = X.shape[0]
        result = np.nan * np.ones(SolutionNumber)

        for jj in range(SolutionNumber):
            self.FE += 1
            if self.FE > self.MaxEvals:
                if np.isinf(self.BestFoundResult):
                    self.BestFoundResult = 1e30
                raise EarlyStop("Raggiunto MaxEvals")

            x = X[jj, :].reshape(-1, 1)
            f = np.nan * np.ones(self.CompNum)

            for k in range(self.CompNum):
                if len(self.RotationMatrix.shape) == 3:
                    rotation_matrix = self.RotationMatrix[:, :, k]
                else:
                    rotation_matrix = self.RotationMatrix

                a = self.transform(
                    (x - self.CompMinPos[k, :].reshape(-1, 1)).T @ rotation_matrix.T,
                    self.Mu[k, :],
                    self.Omega[k, :],
                )
                b = self.transform(
                    rotation_matrix @ (x - self.CompMinPos[k, :].reshape(-1, 1)),
                    self.Mu[k, :],
                    self.Omega[k, :],
                )

                sigma_k = self.CompSigma[k].item()
                lambda_k = self.Lambda[k].item()
                quad = (a @ np.diag(self.CompH[k, :]) @ b).item()
                f[k] = sigma_k + (quad ** lambda_k)

            result[jj] = np.min(f)

            # storia
            self.FEhistory = np.append(self.FEhistory, result[jj])

            # aggiorna best valore e posizione
            if self.BestFoundResult > result[jj]:
                self.BestFoundResult = result[jj]
                # salva la soluzione corrente in forma 1D
                self.BestFoundPosition = x.flatten().copy()

            # controllo AcceptanceThreshold sul best
            error = abs(self.BestFoundResult - self.OptimumValue)
            if error < self.AcceptanceThreshold and np.isinf(self.AcceptanceReachPoint):
                self.AcceptanceReachPoint = self.FE
                raise EarlyStop("Raggiunta AcceptanceThreshold")

        return result

    def transform(self, X, Alpha, Beta):
        Y = X.copy()
        tmp = (X > 0)
        Y[tmp] = np.log(X[tmp])
        Y[tmp] = np.exp(Y[tmp] + Alpha[0] * (np.sin(Beta[0] * Y[tmp]) + np.sin(Beta[1] * Y[tmp])))
        tmp = (X < 0)
        Y[tmp] = np.log(-X[tmp])
        Y[tmp] = -np.exp(Y[tmp] + Alpha[1] * (np.sin(Beta[2] * Y[tmp]) + np.sin(Beta[3] * Y[tmp])))
        return Y


if __name__ == '__main__':
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the folder where you want to read/write files
    folder_path = os.path.join(current_dir)

    # Initialization
    ProblemIndex = 23  # Choose a problem instance range from f1 to f24

    # Preparation and loading of the GNBG parameters based on the chosen problem instance
    if 1 <= ProblemIndex <= 24:
        filename = os.path.join('functions', f'f{ProblemIndex}.mat')
        GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
        MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
        AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
    else:
        raise ValueError('ProblemIndex must be between 1 and 24.')

    print(f"Loaded GNBG problem instance f{ProblemIndex} with dimension {Dimension} and {CompNum} components.\n"
          f"Global optimum value: {OptimumValue} AcceptanceThreshold: {AcceptanceThreshold}, MaxEvals: {MaxEvals}")

    gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)


    # The following code is an example of how a GNBG's problem instance can be solved using an optimizer
    # The Differential Evolution (DE) optimizer is used here as an example. You can replace it with any other optimizer of your choice.

    # Set a random seed for the optimizer
    np.random.seed()  # This uses a system-based source to seed the random number generator

    # Define the bounds for the optimizer based on the bounds of the problem instance
    MULTIPLIER = 100
    lb = -MULTIPLIER*np.ones(Dimension)
    ub = MULTIPLIER*np.ones(Dimension)

    popsize = 50  # population size
    maxiter = MaxEvals // popsize # number of generations/iterations
    #maxiter = 10 #! A SOLO SCOPO DI TEST ___________
    problem = Problem(function=gnbg.fitness, n_var=Dimension, lb=lb, ub=ub)
    try:
        algorithm = ArtificialBeeColony(
            problem=problem, 
            population=popsize, 
            generations=maxiter, 
            seed=42, 
            max_scouts=5, 
            verbose=True
        )
        best_position, best_value = algorithm.run()
       
    except EarlyStop as e:
        print(f"Algoritmo fermato anticipatamente")

    # If you use your own algorithm (not from a library), you can use result = gnbg.fitness(X) to calculate the fitness values of multiple solutions stored in a matrix X.
    # The function returns the fitness values of the solutions in the same order as they are stored in the matrix X.

    # After running the algorithm, the best fitness value is stored in gnbg.BestFoundResult.
    # The best found position is stored in gnbg.BestFoundPosition.
    # The function evaluation number where the algorithm reached the acceptance threshold is stored in gnbg.AcceptanceReachPoint. If the algorithm did not reach the acceptance threshold, it is set to infinity.
    # For visualizing the convergence behavior, the history of the objective values is stored in gnbg.FEhistory, however it needs to be processed as follows:

    convergence = []
    best_error = float('inf')
    for value in gnbg.FEhistory:
        error = abs(value - OptimumValue)
        if error < best_error:
            best_error = error
        convergence.append(best_error)

    print(f"Best found objective value: {gnbg.BestFoundResult} at position {gnbg.BestFoundPosition}")
    print(f"Error from the global optimum: {abs(gnbg.BestFoundResult - OptimumValue)}")
    print(f"Function evaluations to reach acceptance threshold: {gnbg.AcceptanceReachPoint if not np.isinf(gnbg.AcceptanceReachPoint) else 'Not reached'}")
    print("FE usate:", gnbg.FE)
    print("MaxEvals:", gnbg.MaxEvals)
    # Plotting the convergence
    plt.plot(range(1, len(convergence) + 1), convergence)
    plt.xlabel('Function Evaluation Number (FE)')
    plt.ylabel('Error')
    plt.title('Convergence Plot')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.show()