import numpy as np


def choose_seed_nodes(model, budget, vaccination_costs):
    """
    This is a driver functions that takes a threshold model instance assigns nodes to be vaccinated given
    the budget and vaccine costs using the potential-based heuristic.
    :param model: The ndlib diffusion model defined in multiple_contagions.py. The model must already be configured.
    :param budget: The vaccination budget.
    :param vaccination_costs: A list of vaccine costs for each state.
    :return: A dictionary containing the vaccinated nodes, and the contagions that they have been vaccinated against
    """
    node_updates = simulation_run(model)

    potentials = find_potentials(model, node_updates)

    # TODO: Construct a linear program

    # TODO: Call the program

    # TODO: Select the return dictionary from the results

    return None


def simulation_run(model):
    """
    Runs simulation to a fixed point and returns the nodes that move states each time step.
    :param model: The ndlib diffusion model defined in multiple_contagions.py. The model must already be configured.
    :return:
    """
    fixed_point = False
    updated_node_list = []
    while not fixed_point:
        results = model.iteration(node_status=True)
        fixed_point = results['status'] == set()
        updated_node_list.append(results['status'])
    return updated_node_list[:-1]


def find_potentials(model, node_updates):
    """
    Takes simulation results and uses them to along with model's graph to calculate potentials.
    :param model: The ndlib diffusion model defined in multiple_contagions.py. The model must already be configured.
    :param node_updates: The nodes that were updated each time step.
    :return: The potential array.
    """
    # Initialize the 3d-array of potentials to 0.
    potentials = np.zeros(shape=(model.graph.number_of_nodes(), 3, 3), dtype=np.float32)
    # Iterate backwards through the sets starting at the nodes that moved up in state before the fixed point.
    # len(node_updates) - 1 is the end of the list, so we start at -1 of that index.
    T = len(node_updates) - 1
    for i in range(len(node_updates) - 2, 0):
        scaling_factor = (T - i) * (T - i)
        # Retrieve nodes and states from the update dict
        for node, state in node_updates[i]:
            # Update the potential for each state in potentials[node][state]
            for next_step_node, next_step_state in node_updates[i + 1]:
                if next_step_node in model.graph.neighbors(node):
                    if state == 1:
                        potentials[node][state][1] += scaling_factor * (1 + potentials[next_step_node][1][1] +
                                                                        potentials[next_step_node][3][1])
                        potentials[node][state][2] += scaling_factor * potentials[next_step_node][3][2]
                        potentials[node][state][1] += scaling_factor * (1 + potentials[next_step_node][1][3] +
                                                                        potentials[next_step_node][3][3])
                    elif state == 2:
                        potentials[node][state][1] += scaling_factor * potentials[next_step_node][3][1]
                        potentials[node][state][2] += scaling_factor * (1 + potentials[next_step_node][2][2] +
                                                                        potentials[next_step_node][3][2])
                        potentials[node][state][1] += scaling_factor * (1 + potentials[next_step_node][2][3] +
                                                                        potentials[next_step_node][3][3])
                    elif state == 3:
                        potentials[node][state][1] += scaling_factor * (
                                potentials[next_step_node][3][1] + potentials[next_step_node][1][1])
                        potentials[node][state][2] += scaling_factor * (1 + potentials[next_step_node][2][2] +
                                                                        potentials[next_step_node][3][2])
                        potentials[node][state][1] += scaling_factor * (
                                    1 + potentials[next_step_node][1][3] + potentials[next_step_node][2][3] +
                                    potentials[next_step_node][3][3])
    return potentials
