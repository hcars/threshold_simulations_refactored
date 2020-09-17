import numpy as np
import gurobipy as gp
from gurobipy import GRB

def greedy_smc(budget, collection_of_subsets, unsatisfied, requirement_array):
    """
    This provides a ln|X| + 1 approximation to the set mutlicover problem with a budget constraint.
    :param unsatisfied: A list with all the unsatisfied elements
    :param budget: The number of sets that may be chosen.
    :param collection_of_subsets: The collection of available subsets which is a list of lists.
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
        # Check to see if unsatisfied is empty.
        if not unsatisfied:
            break
        i += 1
    return coverage, chosen, unsatisfied


def multi_cover_formulation(available_to_block, next_infected, budget, model, contagion_index):
    subsets = []
    unsatisfied = set()
    # Initialize requirement dict
    requirement_dict = {}
    # Construct subsets and unsatisfied array
    for u in available_to_block:
        subset = set()
        for v in model.graph.neighbors(u):
            if v in next_infected:
                subset.add(v)
                if v not in unsatisfied:
                    unsatisfied.add(v)
        subsets.append(subset)
    # Compute requirement values
    for unsat in unsatisfied:
        threshold = model.params['nodes']["threshold_" + str(contagion_index)][unsat]
        # Find number of infected neighbors
        number_affected = model.graph.nodes[unsat]['affected_' + str(contagion_index)]
        requirement_dict[unsat] = number_affected - threshold + 1
    # Find the cover approximation
    cover_approximation, chosen, unsatisfied_return = greedy_smc(budget, subsets, unsatisfied, requirement_dict)
    return [available_to_block[index] for index in chosen], len(unsatisfied_return)


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


def ilp_formulation(available_to_block, next_infected, budget, model, contagion_index):
    block = {}
    # Initialize requirement dict
    requirement_dict = {}
    # Construct subsets and unsatisfied array
    for u in available_to_block:
        block[u] = []
        for v in model.graph.neighbors(u):
            if v in next_infected:
                block[u].append(v)
                if v not in requirement_dict.keys():
                    threshold = model.params['nodes']["threshold_" + str(contagion_index)][v]
                    # Find number of infected neighbors
                    number_affected = model.graph.nodes[v]['affected_' + str(contagion_index)]
                    requirement_dict[v] = number_affected - threshold + 1
    # Compute requirement values

    next_infected, requirements = gp.multidict(requirement_dict)

    m = gp.Model("smc_ilp")

    block = m.addVars(block.keys(), vtype=GRB.BINARY, name="Blocking node")
    is_covered = m.addVars(next_infected, vtype=GRB.BINARY, name="Is_covered")

    m.addConstrs((gp.quicksum(block[t] for t in block.keys() if r in block[t]) - requirements[r] + 1 <= is_covered[
        r] * len(available_to_block)
                  for r in next_infected), name="Constraint 1")

    m.addConstrs((gp.quicksum(block[t] for t in block.keys() if r in block[t]) - requirements[r] + len(
        available_to_block) >= is_covered[r] * len(available_to_block) for r in next_infected), name="Constraint 2")

    m.addConstr(gp.quicksum(block) <= budget, name="budget")

    m.setObjective(gp.quicksum(is_covered), GRB.MAXIMIZE)

    m.optimize()

    return [int(block[var]) for var in block if int(block[var]) == 1], m.objVal


def try_all_sets(node_infections, budget, model, seed_set, coverage_function=multi_cover_formulation,
                 contagion_index=1):
    """

    :param node_infections:
    :param budget:
    :param model:
    :param seed_set:
    :param coverage_function:
    :param contagion_index:
    :return:
    """
    # Start iteration at i = 1 to find best nodes for contagion contagion_index
    # Int max
    min_unsatisfied = np.iinfo(np.int32).max
    # Stores blocking node ids for contagion_index
    best_solution = []
    # iterations = max(1, len(node_infections) - 1)
    for i in range(len(node_infections) - 1):
        available_to_block = np.setdiff1d(node_infections[i], seed_set)
        if len(available_to_block) <= budget:
            # If we can vaccinate all nodes at infected at this time step return that.
            return available_to_block
        solution, num_unsatisfied = coverage_function(available_to_block, node_infections[i+1], budget, model,
                                                         contagion_index)
        if num_unsatisfied == 0:
            # If we have found an adequate cover, return that.
            return solution
        # Check to see if the solution is the best failing one
        if min_unsatisfied > num_unsatisfied:
            min_unsatisfied = num_unsatisfied
            best_solution = solution
    # If no satisfied set is found, return the one with the least violations
    return best_solution
