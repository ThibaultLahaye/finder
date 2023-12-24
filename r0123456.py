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

@jit(nopython=True)
def cyclic_edge_distance(permutation_1: np.ndarray, permutation_2: np.ndarray) -> np.int64:    
    """
    Calculates the cyclic edge distance between two permutations.

    Cyclic edge distance treats permutations as sets of undirected edges, considering the last element
    connected to the first. For example, the permutation [1, 5, 2, 4, 0, 3] is equivalent to the set of
    edges {(1,5), (5,2), (2,4), (4,0), (0,3), (3,1)}.

    The distance between two permutations is the count of edges that differ. It accounts for undirected
    edges, meaning (i, j) is considered the same as (j, i).

    :param permutation_1: First permutation.
    :param permutation_2: Second permutation.

    :return: Cyclic edge distance between the two permutations.

    Example:
    >>> permutation_1 = np.array([1, 5, 2, 4, 0, 3])
    >>> permutation_2 = np.array([5, 1, 4, 0, 3, 2])
    >>> cyclic_edge_distance(permutation_1, permutation_2)
    3

    Runtime: O(n), where n is the length of the permutations.

    Cyclic edge distance was first described in:
    S. Ronald, "Distance functions for order-based encodings," in Proc. IEEE CEC. IEEE Press, 1997,
    pp. 49–54.

    Author: Vincent A. Cicirello, https://www.cicirello.org/
    """

    count_non_shared_edges = np.int64(0)
    
    # successors2 = np.empty_like(p2)
    # for i in range(successors2.size):
    #     successors2[p2[i]] = p2[(i + 1) % successors2.size]
    successors2 = np.roll(permutation_2, -1)

    for i in range(successors2.size):
        j = (i + 1) % successors2.size
        if permutation_1[j] != successors2[permutation_1[i]] and permutation_1[i] != successors2[permutation_1[j]]:
            count_non_shared_edges += np.int64(1)

    return count_non_shared_edges

@jit(nopython=True)
def cyclic_rtype_distance(permutation_1: np.ndarray, permutation_2: np.ndarray) -> np.int64:
    """
    Calculates the Cyclic RType distance between two permutations.

    Cyclic RType distance treats permutations as sets of directed edges, considering the last element
    connected to the first. For example, the permutation [1, 5, 2, 4, 0, 3] is equivalent to the set of
    directed edges: {(1,5), (5,2), (2,4), (4,0), (0,3), (3,1)}.

    The distance between two permutations is the count of directed edges that differ.

    :param permutation_1: First permutation.
    :param permutation_2: Second permutation.

    :return: Cyclic RType distance between the two permutations.

    Example:
    >>> permutation_1 = np.array([1, 5, 2, 4, 0, 3])
    >>> permutation_2 = np.array([5, 1, 4, 0, 3, 2])
    >>> cyclic_rtype_distance(permutation_1, permutation_2)
    4

    Runtime: O(n), where n is the length of the permutations.

    Cyclic RType distance was introduced in:
    V.A. Cicirello, "The Permutation in a Haystack Problem and the Calculus of Search Landscapes,"
    IEEE Transactions on Evolutionary Computation, 20(3):434-446, June 2016.

    Author: Vincent A. Cicirello, https://www.cicirello.org/
    """

    count_non_shared_edges = np.int64(0)
    
    # successors2 = np.empty_like(p2)
    # for i in range(successors2.size):
    #     successors2[p2[i]] = p2[(i + 1) % successors2.size]
    successors2 = np.roll(permutation_2, -1)

    for i in range(successors2.size):
        j = (i + 1) % successors2.size
        if permutation_1[(i + 1) % successors2.size] != successors2[permutation_1[i]]:
            count_non_shared_edges += np.int64(1)

    return count_non_shared_edges

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