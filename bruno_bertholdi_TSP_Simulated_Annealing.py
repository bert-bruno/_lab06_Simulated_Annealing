'''
Combinatorial Optimization - PPGMNE

Task:

- Define and implement a random search heuristic.
- Define and implement a local search heuristic.
- Define and implement a GRASP algorithm.
- Define and implement a VNS structure.
- Define and implement the Simulated Annealing metaheuristic.

Steps:

1. Download TSP Instances:
    - Head over to the [GITHUB repository](https://github.com/EduPekUfpr/NMUN-7124/tree/main/TSP) and download the TSP instances provided.

---

- Neighborhoods -> //describe neighborhood generation methods//
- Random Search
- Local Search
- GRASP
- VNS
- Simulated Annealing

---

Python 3.12.4

'''

# --- Imports --- #
from dataclasses import dataclass
import pathlib
import pandas as pd
import numpy as np
import random
import time

# --- Defines class for the TSP Problem --- #

class TravellingSalesmanProblem:

    @property
    def get_instances_folder(self):
        ''' Gets the path to the instances folder (folder must be in the same directory as the script).'''
        tsp_name_folder = 'TSP'
        return pathlib.Path().cwd()/tsp_name_folder

    @property
    def create_instances_map(self):
        ''' Creates a dictionary with all instances. '''
        files = []
        for file in self.get_instances_folder.iterdir():
            if file.exists() and str(file).endswith('csv'):
                files.append(file)
        instances_map = {}
        for file in sorted(files):
            instances_map[file.stem] = file
        return self.adjust_instances_map_order(instances_map)
    
    @classmethod
    def adjust_instances_map_order(cls, map_dict: dict):
        ''' Adjust the order of the instances map dictionary changing the second element of the key in a int value.
            - Ex: 'TSP_05_001' -> ['TSP', '05', '001'] -> 5
        '''
        return {
            k: v for k, v
            in sorted(map_dict.items(), key=lambda item: int(item[0].split('_')[1]))
        }
    
    @classmethod
    def read_tsp_file(cls, instance_path):
        ''' Reads the instances file (.csv). '''
        return pd.read_csv(instance_path)
    
    @classmethod
    def calculate_objective_function(cls, tour:list, cities_distance_matrix: np.ndarray, printable_sequence: bool = False) -> float:
        ''' Calculates the objective function. Takes as input a solution (tour) and a distance matrix (cities_distance_matrix).
            - Solution (tour): ordered list containing the cities that the salesman must travel through.
            - Distance matrix (cities_distance_matrix): matrix containing all the distances between cities.
        '''
        n_cities = len(tour) # number of cities to visit. Doesn't count the origin city.
        # --- #
        list_tour_distances = [] # list of all calculated distances between cities.
        for travel in range(n_cities): # each iteration corresponds to a travel
            current_city = tour[travel] # gets the current city
            next_city = tour[(travel + 1) % n_cities] # gets the next city. If the current city is the last, the next city is the first.
            # --- #
            distance_between_cities = cities_distance_matrix[current_city][next_city]
            # --- #
            if printable_sequence:
                # Dictionary containing the distances between cities for rastreability.
                _dict = {f'{current_city} -> {next_city}': float(distance_between_cities)}
                print(_dict) # prints the dictionary with the distance between the cities with the objective to understand the cities' sequence.
            # --- #
            list_tour_distances.appen(distance_between_cities)
        # --- #
        OF = sum(list_tour_distances) # calculates the Objective Function -> sums all distances between selected cities
        return OF
    
    @classmethod
    def check_tsp_constraints(cls, tsp_complete_tour: list, available_cities: list) -> list:
        ''' Checks the constraints of the TSP problem. '''
        # Checks whether the salesman visited all cities or not.
        if not set(tsp_complete_tour) == set(available_cities):
            raise ValueError("Salesman didn't visit all cities yet.")
        # Checks whether the salesman visited each city only once.
        if not len(tsp_complete_tour[:-1]) == len(set(tsp_complete_tour)):
            raise ValueError("Salesman didn't visit each city once.")
        # Checks whether the salesman returned to origin city.
        if not tsp_complete_tour[0] == tsp_complete_tour[-1]:
            raise ValueError("The salesman didn't return to the origin city.")
        # --- #
        return tsp_complete_tour
    
    @classmethod
    def generate_random_solution(cls, cities_distance_matrix: np.ndarray) -> list:
        ''' Randomly generates a solution to the TSP problem. '''
        num_cities = len(cities_distance_matrix) # number of cities to visit, doesn't count the origin city
        base_available_cities = [_ for _ in range(num_cities)] # list of all available cities
        # --- #
        available_cities = base_available_cities.copy() # list of cities that haven't been visited yet -> iterations
        visited_cities = [] # List of visited cities -> iterations
        for _ in range(num_cities): # iterates over each city
            while available_cities: # while there are cities available, continue the loop
                random_index = random.randint(0, len(available_cities) - 1) # chooses a random index from the list of available cities, considering the length of the available cities list after each iteration
                random_visited_city = available_cities[random_index] # chooses a city from the available cities list, based on the random index generated above 
                available_cities.remove(random_visited_city) # removes the random city from the available cities list
                visited_cities.append(random_visited_city) # adds the random city to the visited cities list
        # --- #
        initial_city = visited_cities[0] # identifies the origin city
        tsp_complete_tour = visited_cities + [initial_city] # adds the first city at the add of the tour. Salesman must return back home!
        # --- #
        # Returns the complete tour while checking whether TSP constraints have been met
        return cls.check_tsp_constraints(tsp_complete_tour, base_available_cities)
    
    class RandomSearch:

        @classmethod
        def apply_random_search(cls, cities_distance_matrix: np.ndarray, printable_sequence: bool = False) -> tuple:
            ''' Randomly generates `2*n` solutions and returns the best. '''
            num_cities = len(cities_distance_matrix)
            criteria = 2 * num_cities
            # --- #
            best_of = None
            best_solution = None
            _dict = {} # Results rastreability
            for _ in range(criteria): # iterates over each city
                S = TravellingSalesmanProblem.generate_random_solution(cities_distance_matrix)
                OF = TravellingSalesmanProblem.calculate_objective_function(S, cities_distance_matrix)
                # --- #
                if best_of is None or OF < best_of:
                    best_of = OF
                    best_solution = S
                # --- #
                _dict[f'Iteração {_}'] = {'OF': OF, 'S': S}
            # --- #
            if printable_sequence:
                print(_dict)
            # --- #
            return best_of, best_solution
        
        @classmethod
        def run(cls, save_results=False, return_results=False, printable_sequence=False):
            '''  
            '''
            obj = TravellingSalesmanProblem() # created an instance of the class
            # --- #
            instances = obj.create_instances_map # loads instances to process
            # --- #
            results = {}
            for instance_name, instance_path in instances.items():
                tsp_data = obj.read_tsp_file(instance_path)
                tsp_data.drop(columns=['X', 'Y'], inplace=True)
                tsp_distance = tsp_data.to_numpy()
                # --- Random Search Apply --- #
                s_best, of_best = obj.RandomSearch.apply_random_search(tsp_distance, printable_sequence=printable_sequence)
                # --- Print Random Search Results --- #
                print(f'Instance: {instance_name} | OF_best: {round(of_best, 4)}, S_best: {s_best}')
                # --- #
                results |= {instance_name: {'best_OF': round(of_best, 4), 'best_SOF': s_best}}
            if save_results:
                pd.DataFrame(results).T.to_csv('results.csv')
            if return_results:
                return pd.DataFrame(results).T
            
    class LocalSearch:

        @classmethod
        def generate_neighborhood_by_random_insertion(cls, tour:list, num_neighbors: int):
            ''' Generate a neighborhood by changing the position of the city randomly though subsequent tours. '''
            adjusted_tour = tour[:-1] # removes the last city (must be the same as the first).
            # --- #
            neighbors = []
            n = len(adjusted_tour) # total number of cities (excludes the first and the last).
            for _ in range(num_neighbors):
                remove_index = random.randint(0, n-1) # choose a random city to remove.
                city_to_insert = adjusted_tour[remove_index] # city to insert in a new position
                # --- #
                possible_positions = list(range(1, remove_index)) + list(range(remove_index+1, n)) # possible position to insert the city
                insert_index = random.choice(possible_positions) # chooses a random position to insert the city
                # --- #
                new_tour = adjusted_tour[:remove_index] + adjusted_tour[remove_index+1:] # create a new tour without the city to insert
                new_tour.insert(insert_index, city_to_insert) # insert the city in the new position
                new_tour.append(new_tour[0]) # adds the first city to the end (the salesman must return back home)
                # --- #
                neighbors.append(new_tour)
            return neighbors
        
        @classmethod
        def generate_neighborhood_by_insertion(cls, tour: list, cities_distance_matrix: np.ndarray):
            ''' Generates a neighborhood by changing the position of the city between tours. '''
            # Copies initial solution and assigns it to best tour
            best_tour = tour.copy()
            # Computes the initial solution cost and assigns it to best_of
            best_of = TravellingSalesmanProblem.calculate_objective_function(best_tour, cities_distance_matrix)
            # --- #
            for i in range(len(tour)):
                for j in range(len(tour)):
                    # --- #
                    if i != j:
                        new_tour = tour[:i] + tour[i+1:]
                        new_tour.insert(j, tour[i])
                        new_of = TravellingSalesmanProblem.calculate_objective_function(new_tour, cities_distance_matrix)

                        if new_of < best_of:
                            best_tour = new_tour.copy()
                            best_of = new_of

                        
            return best_tour, best_of

        @classmethod
        def apply_swap_operator(cls, tour: list, cities_distance_matrix: np.ndarray):
            ''' Implements a simple swap local search to improve a solution. '''
            # --- #
            best_tour = tour.copy()
            best_of = TravellingSalesmanProblem.calculate_objective_function(best_tour, cities_distance_matrix)
            # --- #
            improvement = True
            while improvement:
                improvement = False
                num_cities = len(best_tour)
                for i in range(num_cities):
                    for j in range(i + 1, num_cities):
                        new_tour = best_tour[:]
                        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                        new_of = TravellingSalesmanProblem.calculate_objective_function(new_tour, cities_distance_matrix)
                        # --- #
                        if new_of < best_of:
                            best_tour, best_of = new_tour, new_of
                            improvement = True
            return best_tour, best_of
        
        @classmethod
        def apply_local_search(
            cls,
            initial_tour: list,
            cities_distance_matrix: np.ndarray,
            size_criteria: int = None,
            time_criteria: int = None, # in secs
            strategy='first_improvement',
            printable_sequence=False
        ) -> tuple:
            ''' Applies the Local Search method to the TSP problem. '''
            # --- #
            obj = TravellingSalesmanProblem() # creates an instance of the class
            # --- #
            tour = initial_tour.copy() # initial tour through all cities
            OF_initial = obj.calculate_objective_function(tour, cities_distance_matrix) # initial objective function
            OF_chosen = OF_initial # objective function of the chosen neighbor
            # --- #
            _dict = {} # dictionary for rastreability of the results
            # --- First Improvement Strategy --- #
            if strategy == 'first_improvement': # strategy to choose the first improving neighbor
                neighbors = obj.LocalSearch.generate_neighborhood_by_random_insertion(initial_tour, num_neighbors=size_criteria) # generate the neighborhood
                for index, _ in enumerate(neighbors):
                    neighbor = neighbors[index] # gets the neighbor for comparison
                    OF_neighbor = obj.calculate_objective_function(neighbor, cities_distance_matrix) # computes the objective function
                    was_there_improvement = OF_neighbor < OF_chosen # checks whether the neighbor's objective function is better then current.
                    # --- #
                    _dict[f'Iteration {index}'] = {
                        'OF': OF_neighbor,
                        'S': neighbor,
                        'Improvement': was_there_improvement
                    }
                    # --- #
                    if was_there_improvement:
                        tour = neighbor # updates the tour with the neighbor with the best objective function
                        OF_chosen = OF_neighbor # update the objective function of the chosen neighbor
                        break
            # --- Best Improvement Strategy --- #
            elif strategy == 'best_improvement': # strategy to choose the best neighbor that improves the objective function
                start_time = time.time()
                criteria = time_criteria # secs to run
                # --- #
                iteration = len(cities_distance_matrix) # iteration to use as criteria, starts with the number of cities
                while True:
                    current_time = time.time()
                    if (current_time - start_time) > criteria:
                        break
                    neighbors = obj.LocalSearch.generate_neighborhood_by_random_insertion(tour, num_neighbors=iteration) # generate the neighborhood
                    iteration += 1
                    # --- #
                    for index, _ in enumerate(neighbors):
                        neighbor = neighbors[index] # gets tge neighbor for comparison
                        OF_neighbor = obj.calculate_objective_function(neighbor, cities_distance_matrix) # calculates neighbor objective function
                        was_there_improvement = OF_neighbor < OF_chosen # checks whether the neighbor's objective function is better than actual or not.
                        if was_there_improvement:
                            tour = neighbor # updates the tour with the neighbor with the best objective function
                            OF_chosen = OF_neighbor # updates the objective function for the best neighbor
                        # --- #
                        _dict[f'Iteration {index}'] = {
                            'OF': OF_neighbor,
                            'S': neighbor,
                            'Improvement': was_there_improvement
                        }
            # --- #
            if printable_sequence:
                print(_dict)
            # --- #
            S_best = tour.copy()
            OF_best = OF_chosen
            return S_best, OF_best
        
        @classmethod
        def run(
            cls,
            local_search_strategy: str,
            size_criteria = None,
            time_criteria = None, # in secs
            save_results=False,
            return_results=False,
            printable_sequence=False
            ):
            ''' Executes all the steps to solve the TSP problem with Local Search. '''
            # --- #
            obj = TravellingSalesmanProblem() # creates an instance of the class
            # --- #
            instances = obj.create_instances_map # loads all instances to process
            # --- #
            results = {} # random search results
            for instance_name, instance_path in instances.items(): # iterates over each instance and computes the best solution using Random Search.
                tsp_data = obj.read_tsp_file(instance_path) # reads the instance file as DataFrame
                tsp_data.drop(columns=['X', 'Y'], inplace=True) # drops the columns 'X' and 'Y' (not necessary for computation).
                tsp_distance = tsp_data.to_numpy() # convets the DataFrame to a numpy array
                # --- Local Search Apply --- #
                initial_tour = obj.generate_random_solution(tsp_distance) # generate a random initial solution
                s_best, of_best = obj.LocalSearch.apply_local_search(initial_tour, tsp_distance, size_criteria=size_criteria, time_criteria=time_criteria, strategy=local_search_strategy)
                # --- Print Local Search Results --- #
                print(f'local_search_strategy: {local_search_strategy}, Instance: {instance_name} | OF_best: {round(of_best, 4)}, S_best: {s_best}')
                # --- #
                results |= {instance_name: {'best_OF': round(of_best, 4), 'best_SOL': s_best}}
            # --- #
            df = (pd.DataFrame(results)
                  .T
                  .reset_index()
                  .rename({'index': 'instance'}, axis=1)
                  .assign(strategy=local_search_strategy)
                  [['instance', 'strategy', 'best_OF', 'best_SOL']]
                  )
            if save_results:
                df.to_csv(f'results_{local_search_strategy}.csv', index=False)
            if return_results:
                return df


    class GRASP:

        @classmethod
        def create_min_max_distance_matrix(cls, cities_distance_matrix: np.ndarray):
            """ Creates a matrix with the minimum and maximum distances between each city and the others for RCL calculation."""
            n_cities = len(cities_distance_matrix) # number of cities to visit, not counting the home city
            # ---
            for city in range(n_cities): # iterate over each city to identify the minimum and maximum distances for each node
                cities_distance_vector = cities_distance_matrix[city] # get the distance vector of the city to other cities
                non_zero_cities_distance_vector = cities_distance_vector[cities_distance_vector > 0] # remove the zeros from the vector
                # ---
                min_distance = non_zero_cities_distance_vector.min() # get the minimum distance found
                max_distance = non_zero_cities_distance_vector.max() # get the maximum distance found
                yield (min_distance, max_distance) # return the min and max distance found

        @classmethod
        def create_restricted_candidate_list(cls, cities_distance_matrix: np.ndarray, L: float):
            """ Calcultes the Restricted Candidate List (RCL) for each city based on the minimum and maximum distances."""
            min_distances, max_distances = zip(*cls.create_min_max_distance_matrix(cities_distance_matrix))
            # ---
            dict_rcl = {} # creates a dictionary to store RCL's informations
            for node, _ in enumerate(cities_distance_matrix):
                min_distance = min_distances[node] # get the minimum distance for the city
                max_distance = max_distances[node] # get the maximum distance for the city
                # ---
                dict_rcl[f"node_{node}"] = { # store the RCL's informations in the dictionary
                    "rcl": "Random" if max_distance > L else "Greedy",
                    "initial_min_distance": min_distance,
                    "initial_max_distance": max_distance,
                }
            # ---
            return dict_rcl
        
        @classmethod
        def calculate_L(cls, cities_distance_matrix: np.ndarray, alpha: float):
            """ Calcultes the Restricted Candidate List (RCL) for each city based on the minimum and maximum distances.
                - alpha [0, 1]: parameter to control the influence of the maximum distance in the calculation of L.
            """
            distance_matrix_min_distance = cities_distance_matrix.min() # get the minimum distance found in the matrix
            distance_matrix_max_distance = cities_distance_matrix.max() # get the maximum distance found in the matrix
            # ---
            L = distance_matrix_min_distance + alpha * (distance_matrix_max_distance - distance_matrix_min_distance)
            return L
        
        @classmethod
        def identify_cities_with_the_smallest_distance(cls, cities_distance_matrix: np.ndarray) -> list:
            """ Identify the two cities with the smallest distance between them for using as the first part of the tour. """
            mask = cities_distance_matrix > 0 # maks to avoid the diagonal with zeros
            min_distance = np.min(cities_distance_matrix[mask]) # get the minimum distance found in the matrix
            min_distance_node = np.argwhere(cities_distance_matrix == min_distance) # get the index of the minimum distance
            return min_distance_node[0].tolist() # return the cities with the smallest distance between each other

        @classmethod
        def apply_grasp_method(cls, cities_distance_matrix: np.ndarray, alpha: float, printable_sequence: bool = False):
            """ Apply the GRASP method to the TSP problem. """
            matrix = cities_distance_matrix.copy() # copy the distance matrix
            obj = TravellingSalesmanProblem() # create an instance of the class
            # ---
            L = cls.calculate_L(matrix, alpha) # calculate the L parameter
            RCL = cls.create_restricted_candidate_list(matrix, L) # calculate the RCL
            # ---
            n_cities = len(matrix) # number of cities to visit, ignoring the home city
            # ---
            origin_city = None
            destination_city = None
            # ---
            available_cities = [_ for _ in range(n_cities)] # list of all cities available to visit
            visited_cities = [] # list of cities visited by the salesman
            for node in range(n_cities): # iterate over each city, ignoring the initial cities
                if node == 0: # if it is the first node, then identify the two cities with the smallest distance between them to start the tour
                    closest_cities = cls.identify_cities_with_the_smallest_distance(matrix)
                    origin_city, destination_city = closest_cities[0], closest_cities[1] # split the closest cities in origin and destination
                    visited_cities.append(origin_city) # update the list of cities visited by the salesman
                    available_cities.remove(origin_city) # remove the city from the list of available cities
                    # ---
                    matrix[origin_city, :] = np.nan
                    matrix[:, origin_city] = np.nan
                # ---
                elif node > 0:
                    if printable_sequence:
                        print(f"\nBefore: node: {node} | origin_city: {origin_city} | destination_city: {destination_city} | available_cities: {available_cities} | visited_cities: {visited_cities}")
                    # ---
                    origin_city = destination_city
                    # ---
                    node_name = f"node_{origin_city}" # the name of the node to get the RCL
                    origin_city_rcl = RCL[node_name]["rcl"] # get the RCL of the destination city
                    # ---
                    if origin_city_rcl == "Greedy":
                        vector = np.array(matrix[origin_city, :]) # get the row of the destination city
                        mask = (vector != 0) & (~np.isnan(vector)) # mask to avoid zeros and NaN values
                        masked_vector = np.array(vector[mask]) # apply the mask to the vector
                        # ---
                        if masked_vector.size != 0: # if there are cities available to visit other than the last one
                            min_distance = np.min(masked_vector) # get the minimum distance found
                            destination_city = int(np.argwhere(vector == min_distance)[0][0]) # identify the next city based on the minimum distance
                            # ---
                            visited_cities.append(origin_city) # update the list of cities visited by the salesman
                            available_cities.remove(origin_city) # remove the city from the list of available cities
                            # ---
                            matrix[origin_city, :] = np.nan; matrix[:, origin_city] = np.nan # "erase" the row and column of the origin city
                        elif masked_vector.size == 0: # if all possible cities were visited, except the last one
                            visited_cities.append(destination_city) # update the list of cities visited by the salesman
                            available_cities.remove(destination_city) # remove the city from the list of available cities
                    # ---
                    elif origin_city_rcl == "Random":
                        origin_city = destination_city # the new origin city is the destination city
                        visited_cities.append(origin_city) # update the list of cities visited by the salesman
                        available_cities.remove(origin_city) # remove the city from the list of available cities
                        if len(available_cities) > 0: # if there are cities available to visit
                            destination_city = random.choice(available_cities) # the new destination city is randomly chosen from the available cities
                        elif len(available_cities) == 0:
                            destination_city = visited_cities[0] # there aren't cities to randomly choose
                        # ---
                        matrix[origin_city, :] = np.nan; matrix[:, origin_city] = np.nan # "erase" the row and column of the origin city
                    # ---
                    if printable_sequence:
                        print(f"After: node: {node} | RCL: {origin_city_rcl}  | origin_city: {origin_city} | destination_city: {destination_city} | available_cities: {available_cities} | visited_cities: {visited_cities}")
            # ---
            initial_city = visited_cities[0] # identify the first city visited so that the salesman returns at the end of the tour
            S = visited_cities + [initial_city] # add the first city visited at the end of the tour
            OF = obj.calculate_objective_function(S, cities_distance_matrix) # calculate the OF (total distance) of the solution
            return S, OF, L, [v['rcl'] for k, v in RCL.items()]
        
        @classmethod
        def apply_grasp_with_local_search_method(cls, cities_distance_matrix: np.ndarray, alpha: float, time_criteria: int = None, printable_sequence: bool = False):
            """ Apply the GRASP method to the TSP problem. """
            obj = TravellingSalesmanProblem()
            # ---
            S, OF, L, RCL = cls.apply_grasp_method(cities_distance_matrix, alpha=alpha, printable_sequence=printable_sequence)
            # ---
            S, OF = obj.LocalSearch.apply_local_search(S, cities_distance_matrix, time_criteria=time_criteria, strategy='best_improvement')
            return S, OF, L, RCL

        @classmethod
        def run(cls, local_sarch_criteria=None, save_results=False, return_results=False, printable_sequence=False):
            """ Executes all the steps to solve the TSP problem with Random Search. """
            obj = TravellingSalesmanProblem() # create an instance of the class
            # ---
            instances = obj.create_instances_map # load all instances to process
            #--- Criteria: each alpha value is the iteration from 0 to 1, adding .01 in each iteration ---#
            start_point = .0
            lst_alphas = []
            for _ in range(100):
                start_point = round(start_point + .01, 2)
                lst_alphas.append(start_point)
            lst_alphas = list(reversed(lst_alphas))
            # ---
            results = {} # random search results
            for instance_name, instance_path in instances.items(): # iterate over each instance and calculate the best solution using Random Search method
                tsp_data = obj.read_tsp_file(instance_path) # read the instance file as DataFrame
                tsp_data.drop(columns=['X','Y'], inplace=True) # drop the columns 'X' and 'Y' (not necessary for the calculations)
                tsp_distance = tsp_data.to_numpy() # convert the DataFrame to a numpy array
                #--- GRASP Apply ---#
                s_best = None
                of_best = None
                l_best = None
                rcl_best = None
                for alpha in lst_alphas: # 
                    S, OF, L, RCL = obj.GRASP.apply_grasp_method(tsp_distance, alpha=alpha, printable_sequence=printable_sequence) # apply the GRASP Method
                    if of_best is None or of_best < of_best:
                        s_best = S
                        of_best = OF
                        l_best = L
                        rcl_best = RCL
                # --- GRASP with Local Search Apply --- #
                s_best, of_best = obj.LocalSearch.apply_local_search(initial_tour=s_best, cities_distance_matrix=tsp_distance, time_criteria=local_sarch_criteria, strategy='best_improvement')
                # --- Print GRASP/Local Search Results --- #
                print(f"Instance: {instance_name} | OF_best: {round(of_best, 4)}, S_best: {s_best}")
                # ---
                results |= {instance_name: {"best_OF": round(of_best, 4), "best_SOL": s_best, "L_parameter": l_best, "RCL": rcl_best}}
            # ---
            df = (
                    pd.DataFrame(results)
                        .T
                        .reset_index()
                        .rename({"index": "instance"}, axis=1)
                        .assign(method="GRASP")
                        [['instance', 'method', 'best_OF', 'best_SOL', 'L_parameter', 'RCL']]
                )
            if save_results:
                df.drop(columns=['L_parameter', 'RCL']).to_csv(f"results_grasp_with_local_search.csv", index=False)
            if return_results:
                return df

    class VariableNeighborhoodSearch:

        @classmethod
        def apply_grasp_with_vns(
            cls,
            cities_distance_matrix: np.ndarray,
            alpha: float,
            time_criteria: int = None,
            printable_sequence: bool = False
            ):
            """ Apply GRASP and VNS method in the Local Search step to the TSP problem. """
            obj = TravellingSalesmanProblem()
            # Creates the initial solution using the GRASP method
            S, _, _, _ = (
                obj
                    .GRASP()
                    .apply_grasp_method(cities_distance_matrix, alpha=alpha, printable_sequence=printable_sequence)
                )
            # --- Execute the Local Search with VNS, jumping to a local otima to other by the Local Search methods --- #
            initial_tour = S.copy()
            OF_initial = obj.calculate_objective_function(initial_tour, cities_distance_matrix)
            # ---
            OF_chosen = None
            OF_best = None
            S_best = None
            while True:
                if not OF_chosen: # If is the first interation the OF_chosen is the OF_initial
                    OF_chosen = OF_initial
                else: # From the second iteration, run local search methods
                    # --- Eduardo's Local Search Implementation --- #
                    S_local, OF_local = (
                        obj
                            .LocalSearch()
                            .apply_local_search(S, cities_distance_matrix, time_criteria=time_criteria, strategy='best_improvement')
                        )
                    # ---
                    # Check if the local solution is worst than the chosen solution
                    is_local_worst_or_et_than_chosen = OF_local >= OF_chosen
                    # ---
                    if not is_local_worst_or_et_than_chosen: # If it is better, choose the OF
                        OF_chosen = OF_local
                    # ---
                    if is_local_worst_or_et_than_chosen: # If it is worst, jump to another local optima
                        # --- Bruno's Local Search Implementation --- #
                        S_local, OF_local = (
                            obj
                                .LocalSearch()
                                .generate_neighborhood_by_insertion(S_local, cities_distance_matrix)
                            )
                        # ---
                        is_local_better_than_chosen = OF_local < OF_chosen
                        OF_chosen = OF_local
                        if not is_local_better_than_chosen:
                            # OF_best = OF_local
                            # S_best = S_local.copy()
                            # --- Riviane's Local Search Implementation --- #
                            S_best, OF_best = (
                                obj
                                    .LocalSearch()
                                    .apply_swap_operator(route=S_local, distance_matrix=cities_distance_matrix)
                                )
                            break
            
            return S_best, OF_best
        
        @classmethod
        def run(cls, alpha: float, time_criteria: int, save_results=False, return_results=False, printable_sequence=False):
            """ Executes all the steps to solve the TSP problem with Random Search. """
            obj = TravellingSalesmanProblem() # create an instance of the class
            # ---
            instances = obj.create_instances_map # load all instances to process
            # ---
            results = {} # random search results
            for instance_name, instance_path in instances.items(): # iterate over each instance and calculate the best solution using Random Search method
                tsp_data = obj.read_tsp_file(instance_path) # read the instance file as DataFrame
                tsp_data.drop(columns=['X','Y'], inplace=True) # drop the columns 'X' and 'Y' (not necessary for the calculations)
                tsp_distance = tsp_data.to_numpy() # convert the DataFrame to a numpy array
                n_cities = len(tsp_distance) # number of cities to visit, not counting the home city
                # --- VNS: GRASP with Local Search "Jumps" --- #
                # if n_cities >= 150:
                #     time_criteria = n_cities
                s_best, of_best = cls.apply_grasp_with_vns(
                    cities_distance_matrix=tsp_distance,
                    alpha=alpha,
                    time_criteria=time_criteria,
                    printable_sequence=printable_sequence
                )
                # --- Print VNS Results --- #
                print(f"Instance: {instance_name} | OF_best: {round(of_best, 4)}, S_best: {s_best}")
                # ---
                results |= {instance_name: {"best_OF": round(of_best, 4), "best_SOL": s_best}}
             # ---
            df = (
                    pd.DataFrame(results)
                        .T
                        .reset_index()
                        .rename({"index": "instance"}, axis=1)
                        .assign(method="VNS")
                        [['instance', 'method', 'best_OF', 'best_SOL']]
                )
            if save_results:
                df.to_csv(f"results_vns.csv", index=False)
            if return_results:
                return df

    class SimulatedAnnealing:

        @classmethod
        def compute_probability_of_accepting_move(cls, current_of: float, candidate_of: float, temperature: float, sense='minimize') -> float:
            ''' Computes the probability of accepting a candidate objective function.
                - Sense: 'minimize' (standard) sets delta_of to a negative value. 'maximize' sets delta_of to a positive value.
            '''
            # --- #
            maximize_logic_condition = (candidate_of - current_of <= 0) # if sense is set to 'maximize', the difference between candidate objective function and current objective function must be lesser or equal to zero. 
            minimize_logic_condition = (candidate_of - current_of >= 0) # if sense is set to 'minimize', the difference between candidate objective function and current objective function must be greater or equal to zero.
            # --- Validates whether the candidate_of is worse then the current_of, depending on the sense of the optimization problem stated. --- #
            # --- and modifies the probability function depending on the sense of the objective function. --- #
            if sense == 'minimize' and minimize_logic_condition: 
                delta_of = (candidate_of - current_of) * (-1) # computes the difference between the objective function's candidate and current values.
            elif sense == 'maximize' and maximize_logic_condition:
                delta_of = (candidate_of - current_of)
            else:
                raise ValueError('Check the sense of the objective function.') # Lets user know there is something wrong with the input values.
            # --- Computes probability of accepting the candidate solution --- #
            exp = np.exp 
            probability = round(exp(delta_of / temperature), 6)
            # --- #
            return probability
        
        @classmethod
        def linear_cooling_function(cls, alpha:float, initial_temperature: float, iteration:int) -> float:
            ''' Computes the simulated annealing next temperature using a linear strategy.
                - The temperature decreases linearlly and answers the following formula:
                    T_i = T_0 - i * alpha
                - Alpha: cooling rate. Takes a float value between the interval of {0, 1}.
                - T_i: system's temperature in iteration i.
                - T_0: systems's initial temperature.
                - i: iterator.
            '''
            # --- Linear Cooling Function --- #
            delta_temperature = iteration * alpha # computes the value of temperature that will decrease for a given iteration and a given cooling factor.
            temperature_iteration_i = initial_temperature - delta_temperature
            # --- #
            return temperature_iteration_i
        
        @classmethod
        def geometric_cooling_function(cls, alpha:float, initial_temperature:float) -> float:
            ''' Computes the simulated annealing next temperature using the geometric strategy.
                - Temperature decreases given the following formula:
                    T = alpha * T
                - Alpha: takes a value in the ]0, 1[ interval. Popular values for alpha are between
                        0.50 and 0.99.
            '''
            # --- Cooling procedure --- #
            final_temperature = initial_temperature * alpha
            # --- #
            return final_temperature
        
        @classmethod
        def apply_simulated_annealing_method(
            cls,
            sense:str = 'minimize',
            cities_distance_matrix: np.ndarray = None,
            cooling_schedule: str = 'geometric',
            initial_temperature: float = None,
            final_temperature: float = None,
            # alpha: float,
            time_criteria: int = None,
            printable_sequence: bool = False):
            ''' Applies the Simulated Annealing metaheuristics to the TSP problem. '''
            # --- Initialize Travelling Salesman object --- #
            obj = TravellingSalesmanProblem()
            # --- Generates a random initial solution --- #
            (initial_of, initial_solution) = obj.apply_random_search(cities_distance_matrix)
            # --- #
            temperature_criteria = (initial_temperature > final_temperature) # Sets the condition that the initial temperature must be greater than the final temperature.  
            while temperature_criteria: # While the condition is true, the algorithm loops.
                while initial_temperature: # Loops through a given temperature.
                    ### --- Apply swap method to initial solution --- #
                    # insert code here #
                    
                    # --- #

                    break
                break            
    
            pass
        @classmethod
        def run(
            cls, 
            intial_temperature:float, 
            cooling_schedule='geometric', 
            sense='minimize',
            time_criteria: int = None
            ):
            ''' Runs all the steps necessary to perform the Simulated Annealing metaheuristic, applied to a TSP problem. '''
            # --- Initialize Travelling Salesman object --- #
            obj = TravellingSalesmanProblem()
            # --- Load instances --- #
            instances = obj.create_instances_map
            # --- Random Search Results --- #
            results = {}
            # --- #
            pass

            
        
