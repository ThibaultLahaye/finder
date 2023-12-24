import Reporter
import numpy as np
import random
import matplotlib.pyplot as plt
import time

@jit(nopython=True, parallel=False)
def fitness(permutation: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Calculate the cost of a permutation in the Traveling Salesman Problem.

    Checks for np.inf values during each iteration of the loop. 
    If an infinite distance is encountered, the cost is set to np.inf, 
    and the loop is broken out of. This avoids unnecessary computation if 
    a city in the permutation is not connected to the next city.

    The keyword argument 'parallel=True' was specified but no transformation 
    for parallel execution was possible.

    Parameters:
    - distance_matrix (numpy.ndarray): A 2D array representing the distances between cities.
    - permutation (numpy.ndarray): A 1D array representing a permutation of cities.

    Returns:
    - float: The cost of the given permutation.
    """

    num_cities = distance_matrix.shape[0]
    cost = 0.0
    for i in prange(num_cities - 1):
        from_city = permutation[i]
        to_city = permutation[i + 1]

        if np.isinf(distance_matrix[from_city, to_city]):
            cost = np.inf
            break

        cost += distance_matrix[from_city, to_city]

    cost += distance_matrix[permutation[-1], permutation[0]]

    return cost

# Modify the class name to match your student number.
class r0713047:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.

        # Call the reporter with:
        #  - the mean objective function value of the population
        #  - the best objective function value of the population
        #  - a 1D numpy array in the cycle notation containing the best solution
        #    with city numbering starting from 0
        # timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
        # if timeLeft < 0:
        #     break

        # Your code here.

        # Plot results

        return 0

if __name__ == "__main__":
    problem = r0713047()
    problem.optimize("tours/tour50.csv")