import gurobipy as gp
import numpy as np
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

def mycallback(model, where):
    if where == GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(GRB.Callback.PRE_COLDEL)
        rdels = model.cbGet(GRB.Callback.PRE_ROWDEL)
        if cdels or rdels:
            print('%d columns and %d rows are removed' % (cdels, rdels))
    elif where == GRB.Callback.SIMPLEX:
        # Simplex callback
        itcnt = model.cbGet(GRB.Callback.SPX_ITRCNT)
        if itcnt - model._lastiter >= 100:
            model._lastiter = itcnt
            obj = model.cbGet(GRB.Callback.SPX_OBJVAL)
            ispert = model.cbGet(GRB.Callback.SPX_ISPERT)
            pinf = model.cbGet(GRB.Callback.SPX_PRIMINF)
            dinf = model.cbGet(GRB.Callback.SPX_DUALINF)
            if ispert == 0:
                ch = ' '
            elif ispert == 1:
                ch = 'S'
            else:
                ch = 'P'
            print('%d %g%s %g %g' % (int(itcnt), obj, ch, pinf, dinf))
    elif where == GRB.Callback.MIP:
        # General MIP callback
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
        if nodecnt - model._lastnode >= 100:
            model._lastnode = nodecnt
            actnodes = model.cbGet(GRB.Callback.MIP_NODLFT)
            itcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
            print('%d %d %d %g %g %d %d' % (nodecnt, actnodes,
                  itcnt, objbst, objbnd, solcnt, cutcnt))
        if abs(objbst - objbnd) < 0.025 * (1.0 + abs(objbst)):
            print('Stop early - 5% gap achieved')
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        x = model.cbGetSolution(model._vars)
        print('**** New solution at node %d, obj %g, sol %d, '
              'x[0] = %g ****' % (nodecnt, obj, solcnt, x[0]))
    elif where == GRB.Callback.MIPNODE:
        # MIP node callback
        print('**** New node ****')
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x = model.cbGetNodeRel(model._vars)
            model.cbSetSolution(model.getVars(), x)
    elif where == GRB.Callback.BARRIER:
        # Barrier callback
        itcnt = model.cbGet(GRB.Callback.BARRIER_ITRCNT)
        primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
        dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
        priminf = model.cbGet(GRB.Callback.BARRIER_PRIMINF)
        dualinf = model.cbGet(GRB.Callback.BARRIER_DUALINF)
        cmpl = model.cbGet(GRB.Callback.BARRIER_COMPL)
        print('%d %g %g %g %g %g' % (itcnt, primobj, dualobj,
              priminf, dualinf, cmpl))
    elif where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        model._logfile.write(msg)


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

    next_infected = list(requirement_dict.keys())
    m = gp.Model("smc_ilp")

    blocking_vars = m.addVars(block.keys(), vtype=GRB.BINARY, name="Blocking node")
    is_covered = m.addVars(next_infected, vtype=GRB.BINARY, name="Is_covered")

    m.addConstrs(
        (gp.quicksum(blocking_vars[t] for t in block.keys() if r in block[t]) - requirement_dict[r] + 1 <= is_covered[
            r] * len(available_to_block)
         for r in next_infected), name="Constraint 1")

    m.addConstrs((gp.quicksum(blocking_vars[t] for t in block.keys() if r in block[t]) - requirement_dict[r] + len(
        available_to_block) >= is_covered[r] * len(available_to_block) for r in next_infected), name="Constraint 2")

    m.addConstr(gp.quicksum(blocking_vars) <= budget, name="budget")

    m.setObjective(gp.quicksum(is_covered), GRB.MAXIMIZE)

    m.optimize(mycallback)

    solution = [var for var in blocking_vars if blocking_vars[var].x == 1.0]

    return solution, m.objVal


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
        solution, num_unsatisfied = coverage_function(available_to_block, node_infections[i + 1], budget, model,
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








