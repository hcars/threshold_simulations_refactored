import numpy as np


def greedy_smc(budget, collection_of_subsets, unsatisfied, requirement_array):
    """
    This provides a ln|X| + 1 approximation to the set mutlicover problem with a budget constraint.
    :param unsatisfied: A list with all the unsatisfied elements
    :param budget: The number of sets that may be chosen.
    :param collection_of_subsets: The collection of available subsets which is a list of sets.
    :param requirement_array: A dict containing the coverage requirement for each node.
    :return: A coverage, all of the sets that are chosen, and the unsatisfied set.
    """
    # A list to store chosen subsets
    coverage = []
    # An set to store if a set is chosen
    chosen = set()
    i = 0
    while i < budget:
        # Iterate over unchosen sets and check the size of their intersection with the chosen set.
        max_size = 0
        max_index = 0
        for j in range(len(collection_of_subsets)):
            if j not in chosen:
                curr_size = len(unsatisfied.intersection(collection_of_subsets[j]))
                if curr_size > max_size:
                    max_size = curr_size
                    max_index = j
        # print(max_index, max_size)
        # Indicate that the max intersect set is chosen
        chosen.add(max_index)
        chosen_set = collection_of_subsets[max_index]
        coverage.append(chosen_set)
        # Decrement coverage requirements and remove from the unsatisfied set as necessary
        for element in chosen_set:
            requirement_array[element] -= 1
            if requirement_array[element] == 0:
                unsatisfied.remove(element)
        if not unsatisfied:
            break
        i += 1
    return coverage, chosen, unsatisfied


def coverage_heuristic(budget_1, budget_2, model):
    """
    This drives the method drives the contagion blocking and details can be found in the paper.
    :param budget_1: The budget for contagion 1.
    :param budget_2: The budget for contagion 2.
    :param model: The model with its underlying graph.
    :return: A choice of nocdes to block.
    """
    node_infections_1, node_infections_2, results = model.simulation_run()
    # Run through the CBH from DMKD for both contagions.
    choices_1 = try_all_sets(node_infections_1, budget_1, model, 1)
    choices_2 = try_all_sets(node_infections_2, budget_2, model, 2)
    # Return the choices found.
    return choices_1, choices_2


def try_all_sets(node_infections, budget, model, seed_set, threshold_index=1):
    # Start iteration at i = 1 to find best nodes for contagion threshold_index
    min_unsatisfied = np.iinfo(np.int32).max
    best_solution = []
    iterations = max(1, len(node_infections) - 1)
    for i in range(iterations):
        available_to_block = node_infections[i].difference(seed_set)
        if len(node_infections[i]) <= budget:
            # If we can vaccinate all nodes at infected at this time step return that.
            return node_infections[i]
        subsets = []
        unsatisfied = set()
        # Initialize requirement dict
        requirement_array = {}
        # Construct subsets and unsatisfied array
        for u in node_infections[i]:
            subset = set()
            for v in model.graph.neighbors(u):
                if v in node_infections[i + 1]:
                    subset.add(v)
                    if v not in unsatisfied:
                        unsatisfied.add(v)
            subsets.append(subset)
        # Compute requirement values
        for unsat in unsatisfied:
            threshold = model.params['nodes']["threshold_" + str(threshold_index)][unsat]
            # Find number of infected neighbors
            number_affected = model.graph.nodes[unsat]['affected_' + str(threshold_index)]
            requirement_array[unsat] = number_affected - threshold + 1
        # Find the cover approximation
        cover_approximation, chosen, unsatisfied_return = greedy_smc(budget, subsets, unsatisfied, requirement_array)
        if not unsatisfied_return:
            # If we have found an adequate cover, return that.
            return [node_infections[i][index] for index in chosen]
        # Check to see if the solution is the best failing one
        num_unsatisfied = len(unsatisfied_return)
        if min_unsatisfied > num_unsatisfied:
            min_unsatisfied = num_unsatisfied
            best_solution = [node_infections[i][index] for index in chosen]
    # If no satisfied set is found, return the one with the least violations
    return best_solution
