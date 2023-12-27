""" General Problem Description
This file contains the implementation of an evolutionary algorithm for the Traveling Salesman Problem (TSP).
A distance matrix is read from a tour[length].cvs file, where length is the number of cities in the problem.
The distance matrix is a 2D array representing the distances between cities. Unreachable cities are represented
by np.inf values. 
"""

""" Still to Complete
TODO: Complete the implementation of the different recombination operators.
TODO: Complete the implementation of the main optimization loop.
"""

""" Notes

# Caching
Testing showed that fitness_cached() is slower than fitness() for the given tour files.
Even when using rotate_0_up_front() to reduce the number of permutations that need to be cached.
This was mainly due to a low hit rate of the cache, for larger tour files (1000).
Even when evaluating a 
"""
import Reporter
import numpy as np
from numba import jit
import time # For testing

# Utility Functions 

@jit(nopython=True)
def rotate_0_up_front(permutation: np.ndarray) -> np.ndarray:
    idx = np.argmax(permutation == 0)
    return np.roll(permutation, -idx)

@jit(nopython=True)
def best(fitness_values: np.ndarray) -> np.float64:
    return np.min(fitness_values)

@jit(nopython=True)
def mean(fitness_values: np.ndarray) -> np.float64:
    mask = np.isfinite(fitness_values)
    return np.mean(fitness_values[mask])

# Permutation fitness function

@jit(nopython=True)
def fitness(permutation: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Calculate the cost of a permutation in the Traveling Salesman Problem.

    First, the cache is checked for the cost of the given permutation.
    If the cost is not cached, the cost is calculated, returned and cached.

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
    for i in range(num_cities - 1):
        from_city = permutation[i]
        to_city = permutation[i + 1]

        if np.isinf(distance_matrix[from_city, to_city]):
            cost = np.inf
            break

        cost += distance_matrix[from_city, to_city]

    cost += distance_matrix[permutation[-1], permutation[0]]

    return cost

@jit(nopython=True)
def fitnesses(population: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    pop_size = population.shape[0]

    fitness_values = np.empty(pop_size, dtype=np.float64)

    for i in np.arange(pop_size):
        fitness_values[i] = fitness(population[i], distance_matrix)

    return fitness_values

# Distance Functions

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

# Initialization Operators

@jit(nopython=True)
def random_permutation(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate a random permutation of cities.

    Parameters:
    - distance_matrix (np.ndarray): A 2D array representing the distances between cities.

    Returns:
    - np.ndarray: A 1D array representing a permutation of cities.
    """

    num_cities = distance_matrix.shape[0]
    permutation = np.random.permutation(num_cities)

    return permutation

@jit(nopython=True)
def valid_permutation(distance_matrix: np.ndarray) -> np.ndarray:
    permutation = np.empty(distance_matrix.shape[0], dtype=np.int64)
    cities = np.arange(distance_matrix.shape[0])

    attempts = 0
    while attempts < 5:

        permutation[0] = np.random.randint(0, distance_matrix.shape[0])
        for i in range(1, distance_matrix.shape[0]):
            
            neighbours_mask = np.isfinite(distance_matrix[permutation[i-1]]) #
            neighbours_mask[permutation[:i]] = False # 
            neighbours = cities[neighbours_mask] # 

            # check if no neighbours left
            if neighbours.size == 0:
                print("no neighbours left")
                attempts += 1
                break

            permutation[i] = np.random.choice(neighbours) # select random neighbour

            # check if last city is connected to first city
            if i == distance_matrix.shape[0] - 1:
                distance = distance_matrix[permutation[i], permutation[0]]
                if distance == np.inf:
                    print("last city not connected to first city")
                    attempts += 1
                    break 
        
        else: 
            return permutation

    # Last resort: return random permutation
    print("last resort")
    return np.random.permutation(distance_matrix.shape[0])

@jit(nopython=True)
def greedy_permutation(distance_matrix: np.ndarray) -> np.ndarray:
    permutation = np.empty(distance_matrix.shape[0], dtype=np.int64)
    cities = np.arange(distance_matrix.shape[0])

    attempts = 0
    while attempts < 5:
        permutation[0] = np.random.randint(0, distance_matrix.shape[0])
        for i in range(1, distance_matrix.shape[0]):
            
            prev_city = permutation[i-1]

            neighbours_mask = np.isfinite(distance_matrix[prev_city])
            neighbours_mask[permutation[:i]] = False
            neighbours = cities[neighbours_mask]

            if neighbours.size == 0:
                break

            neighbours_r_idx = np.argmin(distance_matrix[prev_city][neighbours])

            permutation[i] = neighbours[neighbours_r_idx]

            if i == distance_matrix.shape[0] - 1:
                distance = distance_matrix[permutation[i], permutation[0]]
                if distance == np.inf:
                    break
        
        else: 
            return permutation

    # Last resort: return random permutation
    print("last resort")
    return np.random.permutation(distance_matrix.shape[0])
        
@jit(nopython=True)
def initialize_population(distance_matrix: np.ndarray, lambda_: np.int64) -> np.int64: 
    
    random_number = int(lambda_*0.05)
    greedy_number = int(lambda_*0.15)
    valid_number  = int(lambda_*0.80)

    assert(random_number + greedy_number + valid_number == lambda_)

    population = np.empty((lambda_, distance_matrix.shape[0]), dtype=np.int64)
    
    idx = np.random.permutation(lambda_)

    for r in np.arange(0, random_number):
        population[idx[r]] = random_permutation(distance_matrix)
    
    for g in np.arange(random_number, greedy_number):
        population[idx[g]] = greedy_permutation(distance_matrix)

    for v in np.arange(greedy_number, lambda_):
        population[idx[v]] = valid_permutation(distance_matrix)

    return population

# Selection Operators

@jit(nopython=True, parallel=True)
def k_tournament_selection (population: np.ndarray, fitness_values: np.ndarray, k:np.int64) -> np.ndarray:
    """
    Perform k-tournament selection on a population based on their fitness values.

    Parameters:
    - population (np.ndarray): 2D array representing the population, where each row is an individual.
    - fitness_values (np.ndarray): 1D array containing the fitness values corresponding to each individual in the population.
    - k (np.int64): The size of the tournament.

    Returns:
    - np.ndarray: 1D array representing the selected individual with the highest fitness in the tournament.

    Tournament Selection Process:
    Randomly select 'k' individuals from the population.
    The individual with the highest fitness among the selected individuals is chosen as the winner.

    Example:
    Population:
    [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10],
     [11, 12, 13, 14, 15],
     [16, 17, 18, 19, 20]]

    Fitness Values: [20, 15, 18, 12]

    Tournament Selection with k=3:
    Randomly selected indices: [2, 0, 1]

    Selected Individuals:
    [[11, 12, 13, 14, 15],
     [1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10]]

    Winner: [11, 12, 13, 14, 15]
    """

    tournament_indices = np.random.choice(population.shape[0], k, replace=False)
    winner_permutation = population[np.argmax(fitness_values[tournament_indices])]
    
    return winner_permutation

# Mutation Operators

@jit(nopython=False)
def inversion_mutation(permutation: np.ndarray, alpha_: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """
    Perform inversion mutation on a permutation array within a specified window.

    Parameters:
    - permutation (numpy.ndarray): The input permutation array.
    - start (numpy.int64): The starting index of the mutation window.
    - end (numpy.int64): The ending index of the mutation window.
    - alpha_ (numpy.float64): The probability of performing the inversion mutation.

    Returns:
    - numpy.ndarray: The mutated permutation array.

    ASCII Art Visualization:
    
    Original Permutation: [1, 2, 3, 4, 5, 6, 7, 8, 9]

    Mutation Window:      |-------[        ]--------|
                              start        end

    Mutation Process:
    If random probability (alpha_) is met, a subset within the window is inverted.

    Example:
    Original Permutation: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    Mutation Window:      |-------[        ]--------|
                            start           end

    Subset to be inverted:         [4, 5, 6]

    Mutated Permutation: [1, 2, 3, 6, 5, 4, 7, 8, 9]
    """
    
    if np.random.random() <= alpha_:
        num_cities = distance_matrix.shape[0]

        start_idx = np.random.randint(0, num_cities)
        size = np.random.randint(2, num_cities)

        idx = np.arange(start_idx, start_idx + size)
        subset = permutation[idx % num_cities]

        mutated_permutation = permutation.copy()
        mutated_permutation[idx % num_cities]  = subset[::-1]
                        
        return mutated_permutation
    
    return permutation

@jit(nopython=True)
def scramble_mutation(permutation: np.ndarray, alpha_: np.float64, distance_matrix: np.ndarray) -> np.ndarray:
    """Scramble mutation: randomly choose 2 indices and scramble that subsequence."""   

    if random.random() < alpha_:
        i = random.randint(0, permutation.size - 1)
        j = random.randint(0, permutation.size - 1)
        if j < i:
            i, j = j, i
        new_order = np.copy(permutation)
        np.random.shuffle(new_order[i:j])
        return new_order

    return permutation

# Crossover Operators

@jit(nopython=True)
def k_point_crossover(parent_1: np.ndarray, parent_2: np.ndarray, k: np.int64) -> np.ndarray: #TODO
    pass

@jit(nopython=True)
def uniform_crossover(parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray:
    """
    https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#:~:text=k%20crossover%20points.-,Uniform%20crossover,-%5Bedit%5D
    """
    assert len(parent1) == len(parent2)

    child_genome = np.empty_like(parent1)

    # Randomly select genes from each parent with equal probability
    for i in range(len(parent1)):
        child_genome[i] = np.random.choice([parent1[i], parent2[i]])

    return child_genome

@jit(nopython=True)
def pmx_crossover(parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray: #TODO
    pass

@jit(nopython=True)
def ox_crossover(parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray: #Testing
    """
    A random subset of the parent_1 is copied to the child. The remaining positions are filled
    with the values from parent_2 that are not already in the child. The values are copied from
    parent_2 in the same order as they appear in parent_2.
    This inplementation can also handle wrap-around, meaning that the subset can start at the
    last index of parent_1 and wrap around to the first index.  
    """
    assert parent_1.size == parent_2.size

    num_cities = parent_1.size

    start_idx = np.random.randint(0, num_cities)
    size = np.random.randint(0, num_cities)

    idx = np.arange(start_idx, start_idx + size)

    child = np.full(num_cities , fill_value=-1)

    child[idx % num_cities] = parent_1[idx % num_cities]
    child[child == -1] = [i for i in parent_2 if i not in child]

    return child

@jit(nopython=True)
def edge_assembly_crossover(parent_1: np.ndarray, parent_2: np.ndarray) -> np.ndarray: #TODO
    pass

# Local Search Operators

@jit(nopython=True)
def two_opt_local_search(permutation: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    num_cities = distance_matrix.shape[0]

    new_permutation = permutation.copy()
    new_permutation_fitness = fitness(permutation, distance_matrix)

    for a in np.arange(0, num_cities):
        for b in np.arange(a+1, num_cities):
            candidate_permutation = np.concatenate((new_permutation[:a], new_permutation[a:b][::-1], new_permutation[b:]))
            candidate_permutation_fitness = fitness(candidate_permutation, distance_matrix) 

            if candidate_permutation_fitness < new_permutation_fitness:
                new_permutation = candidate_permutation
                new_permutation_fitness = candidate_permutation_fitness

    return new_permutation

@jit(nopython=True)
def two_opt_local_search_pop(population: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:

    new_population = np.empty_like(population)

    for i in np.arange(0, population.shape[0]):
        print(i)
        new_population[i] = two_opt_local_search(population[i], distance_matrix)

    return new_population

# Fitness Sharing

@jit(nopython=True)
def fitness_sharing(fitness_values: np.ndarray, 
                    population: np.ndarray,
                    shape_: np.float64, sigma_: np.float64) -> np.ndarray: #Testing
    """
    This inplementation leverages the fact that the cyclic r-type distance between two permutations
    is symmetric. This means that the distance between permutation_1 and permutation_2 is the same and 
    dist_matrix[i][j] equals dist_matrix[j][i].

    population (numpy.ndarray): A 2D matrix representing the population and offspring.
    fitness_values (numpy.ndarray): A 1D array containing the fitness values of the population and offspring.
    """
    
    num_permutations = population.shape[0]

    dist_matrix = np.empty((num_permutations, num_permutations), dtype=np.float64)
    shared_fitness_values = np.empty(num_permutations, dtype=np.float64)

    for i in np.arange(0, num_permutations):
        for j in np.arange(0, num_permutations):

            if j >= i: 
                dist_matrix[i, j] = cyclic_rtype_distance(population[i], population[j])

                if dist_matrix[i, j] <= sigma_:
                    shared_fitness_values[i] = fitness_values[i]*(1 - (dist_matrix[i, j] / sigma_)**shape_)
                else:
                    shared_fitness_values[i] = fitness_values[i]

            else:
                if dist_matrix[j, i] <= sigma_:
                    shared_fitness_values[i] = fitness_values[i]*(1 - (dist_matrix[j, i] / sigma_)**shape_)
                else:
                    shared_fitness_values[i] = fitness_values[i]

    print(shared_fitness_values)
    return shared_fitness_values

# Elimination Operators

@jit(nopython=True)
def lambda_mu_elimination(population_and_offspring: np.ndarray, fitness_values: np.ndarray, shared_fitness_values: np.ndarray, lamda_: np.int64) -> np.ndarray:
    """
    Order the rows of the population_and_offspring matrix based on the corresponding values in the
    shared_fitness_values array and return the top lamda_ rows along with their shared fitness values.

    Parameters:
    - population_and_offspring (numpy.ndarray): A 2D matrix representing the population and offspring.
    - shared_fitness_values (numpy.ndarray): A 1D array containing shared fitness values corresponding to the rows.
    - lamda_ (numpy.int64): The number of best values to keep.

    Returns:
    tuple: A tuple containing two elements:
        - numpy.ndarray: The top lamda_ rows of the ordered population_and_offspring matrix.
        - numpy.ndarray: The corresponding shared fitness values for the selected population.
    """
    sorted_indices = np.argsort(shared_fitness_values)

    selected_population = population_and_offspring[sorted_indices][0:lamda_]
    selected_fitness_values = fitness_values[sorted_indices][0:lamda_]

    return selected_population, selected_fitness_values

# Modify the class name to match your student number.
class r0713047:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):

        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        num_cities = distance_matrix.shape[0]

        ################ PARAMETERS ################
        lambda_                   = np.int64(100)
        mu_                       = np.int64(50)
        k_                        = np.int64(3)
        mutation_alpha_           = np.float64(0.35)
        fitness_sharing_sigma_    = np.float64(0.35*num_cities)
        fitness_sharing_alpha_    = np.float64(1)
        ############################################

        # initialize population
        population = initialize_population(distance_matrix, lambda_)
        population = two_opt_local_search_pop(population, distance_matrix)

        # Evaluate the fitness of the initial population.
        fitness_values = fitnesses(population, distance_matrix)

        # Mainloop
        while True: 
        
            offspring = np.empty((mu_, distance_matrix.shape[0]), dtype=np.int64)
            offspring_fitness_values = np.empty(mu_, dtype=np.float64)
            for i in np.arange(0, mu_):
                # Select parents from the population
                print("Selecting parents")
                parent_1 = k_tournament_selection(population, fitness_values, k_)
                parent_2 = k_tournament_selection(population, fitness_values, k_)

                # Perform crossover on the parents
                print("Performing crossover")
                child = ox_crossover(parent_1, parent_2)

                # Perform mutation on the child
                print("Performing mutation")
                child = inversion_mutation(child, mutation_alpha_, distance_matrix)

                # Add the child to the offspring
                offspring[i] = child

                # Evaluate the fitness of the child
                offspring_fitness_values[i] = fitness(child, distance_matrix)
            
            # Combine the population and offspring
            print("Combining population and offspring")
            population_and_offspring = np.concatenate((population, offspring))
            population_and_offspring_fitness_values = np.concatenate((fitness_values, offspring_fitness_values))

            # Perform fitness sharing on the combined population and offspring
            print("Performing fitness sharing")
            shared_fitness_values = fitness_sharing(population_and_offspring_fitness_values, population_and_offspring, fitness_sharing_alpha_, fitness_sharing_sigma_)

            # Select the best lambda_ individuals from the combined population and offspring according to their shared fitness values
            print("Selecting best individuals")
            population, fitness_values = lambda_mu_elimination(population_and_offspring, fitness_values, shared_fitness_values, lambda_)

            # Report results
            meanObjective = mean(fitness_values)
            bestObjective = best(fitness_values)
            bestSolution = rotate_0_up_front(population[0])
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            assert bestObjective == fitness(bestSolution, distance_matrix)

            if timeLeft < 2:
                break

        # Your code here.

        # Plot results

        return 0

if __name__ == "__main__":
    problem = r0713047()
    problem.optimize("tours/tour50.csv")