import csv
from os.path import exists

import networkx as nx
import numpy as np

import coverage_heuristic as cbh
import utils


def choose_seed(core, seed_size):
    component = np.random.choice(core, seed_size, replace=False)
    seed_set_1 = []
    seed_set_2 = []
    seed_set_3 = []
    for index in range(len(component)):
        roll = np.random.randint(1, 4)
        if roll == 3:
            seed_set_3.append(component[index])
        elif roll == 2:
            seed_set_2.append(component[index])
        elif roll == 1:
            seed_set_1.append(component[index])
        else:
            raise ValueError("Not in the correct range")
    return seed_set_1, seed_set_2, seed_set_3


def choose_nodes_by_degree(G, budget_1, budget_2, seed_set):
    choices_1 = []
    choices_2 = []
    nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    index = 0
    while len(choices_1) < budget_1:
        if nodes_by_degree[index][0] not in seed_set:
            choices_1.append(nodes_by_degree[index][0])
        index += 1
    index = 0
    while len(choices_2) < budget_2:
        if nodes_by_degree[index][0] not in seed_set:
            choices_2.append(nodes_by_degree[index][0])
        index += 1
    return choices_1, choices_2


def choose_randomly(G, budget_1, budget_2, seed_set):
    nodes = list(G.nodes())
    choices_1 = []
    choices_2 = []
    index = np.random.randint(0, len(nodes))
    while len(choices_1) < budget_1:
        if nodes[index] not in seed_set and nodes[index] not in choices_1:
            choices_1.append(nodes[index])
        index = np.random.randint(0, len(nodes))
    index = np.random.randint(0, len(nodes))
    while len(choices_2) < budget_2:
        if nodes[index] not in seed_set and nodes[index] not in choices_2:
            choices_2.append(nodes[index])
        index = np.random.randint(0, len(nodes))
    return choices_1, choices_2


def main():
    field_names = ['network_name', 'threshold', 'seed_size', 'budget_total']
    # Add fields for node counts for each blocking method
    for blocking in ["_no_block", "_cbh", "_degree", "_random"]:
        field_names += [
            str(i) + blocking for i in
            range(4)]
    with open('complex_net_proposal/experiment_results/astroph_results.csv', 'w', newline='') as csv_fp:
        csv_writer = csv.writer(csv_fp, delimiter=',')
        csv_writer.writerow(field_names)
    # Load in networks
    network_folder = "complex_net_proposal/experiment_networks/"
    # Constants for stochastic portion do not change
    seeds = (6893, 20591, 20653)
    net_names = ["astroph"]
    thresholds = (2, 3, 4)
    budgets = [.005] + [.01 + i * .01 for i in range(12)]
    sample_number = 10

    for i in range(len(net_names)):
        np.random.seed(seeds[i])
        net_name = net_names[i]
        G = nx.read_edgelist(network_folder + net_name + '.edges', nodetype=int, create_using=nx.Graph)
        # If there is a node file add it.
        if exists(net_name + ".nodes"):
            with open(network_folder + net_name + ".nodes", 'r') as node_fp:
                node_str = node_fp.readline().strip('\n')
                node_id = int(node_str)
                G.add_node(node_id)
        for node in G.nodes:
            G.nodes[node]['affected_1'] = 0
            G.nodes[node]['affected_2'] = 0
        # Select k-core
        k_core = list(nx.k_core(G, 20).nodes())[0]
        connected_core = [k_core]
        while len(connected_core) < 200:
            index_to_expand = np.random.randint(0, len(connected_core))
            choose_from = list(nx.neighbors(G, connected_core[index_to_expand]))
            choice = choose_from[np.random.randint(0, len(choose_from))]
            if choice not in connected_core:
                connected_core.append(choice)
        for seed_size in [20]:
            # Initialize accumulators Mult-level dict threshold -> (budget -> (results_avg, results_blocked_avg,
            # results_degree_avg, results_random))
            avgs = {
                threshold: {int(budget * G.number_of_nodes()): tuple({state: 0 for state in range(4)} for i in range(4))
                            for budget in budgets} for threshold in
                thresholds}
            for sample in range(sample_number):
                # Choose seed set
                seed_set_1, seed_set_2, seed_set_3 = choose_seed(connected_core, seed_size)
                seed_set = set(seed_set_1 + seed_set_2 + seed_set_3)
                for k in range(len(thresholds)):
                    # Pull out threshold
                    threshold = thresholds[k]
                    for j in range(len(budgets)):
                        # Get the budget
                        budget = int(budgets[j] * G.number_of_nodes())
                        # Configure model
                        model = utils.config_model(G, threshold, seed_set_1, seed_set_2, seed_set_3)
                        node_infections_1, node_infections_2, results = model.simulation_run()
                        # Analyze node counts
                        infected_1 = results['node_count'][1] + results['node_count'][3]
                        infected_2 = results['node_count'][2] + results['node_count'][3]
                        total_infected = sum(results['node_count'][i] for i in range(1, 4))
                        # Select nodes appropriately
                        ratio_infected_1 = infected_1 / total_infected
                        budget_1 = int(ratio_infected_1 * budget)
                        budget_2 = budget - budget_1
                        # Run through the CBH from DMKD for both contagions.
                        choices_1 = cbh.try_all_sets(node_infections_1, budget_1, model, set(seed_set_1 + seed_set_3),
                                                     1)
                        if len(choices_1) < budget_1:
                            budget_2 += budget_1 - len(choices_1)
                        choices_2 = cbh.try_all_sets(node_infections_2, budget_2, model, set(seed_set_2 + seed_set_3),
                                                     2)
                        if len(choices_2) < budget_2:
                            choices_1 = cbh.try_all_sets(node_infections_1, budget_1 + (budget_2 - len(choices_2)),
                                                         model, set(seed_set_1 + seed_set_3),
                                                         1)

                        # TODO: Think about the situation where we can block both at a certain time steps.

                        # Run again with the CBH blocking
                        # Configure model
                        model = utils.config_model(G, threshold, seed_set_1, seed_set_2, seed_set_3, choices_1,
                                                   choices_2)
                        results_blocked = model.simulation_run(first_infected=False)
                        # Find high degree nodes
                        choices_1, choices_2 = choose_nodes_by_degree(G, budget_1, budget_2, seed_set)
                        # Run forward
                        model = utils.config_model(G, threshold, seed_set_1, seed_set_2, seed_set_3, choices_1,
                                                   choices_2)
                        results_blocked_degree = model.simulation_run(first_infected=False)
                        # Find random nodes
                        choices_1, choices_2 = choose_randomly(G, budget_1, budget_2, seed_set)
                        # Run forward
                        model = utils.config_model(G, threshold, seed_set_1, seed_set_2, seed_set_3, choices_1,
                                                   choices_2)
                        results_random = model.simulation_run(first_infected=False)
                        for state in range(4):
                            avgs[threshold][budget][0][state] += results['node_count'][state]
                            avgs[threshold][budget][1][state] += results_blocked['node_count'][state]
                            avgs[threshold][budget][2][state] += results_blocked_degree['node_count'][state]
                            avgs[threshold][budget][3][state] += results_random['node_count'][state]
            for threshold in thresholds:
                for budget in budgets:
                    budget = int(budget * G.number_of_nodes())
                    for accumulator in range(4):
                        for state in range(4):
                            avgs[threshold][budget][accumulator][state] /= sample_number
                    with open('complex_net_proposal/experiment_results/astroph_results.csv', 'a', newline='') as results_fp:
                        csv_writer = csv.writer(results_fp, delimiter=',')
                        # Write problem data
                        result_data = [net_name, str(threshold), str(seed_size),
                                       str(budget)]
                        # Add in the averages
                        for accumulator in range(4):
                            result_data += list(
                                map(lambda x: str(x), avgs[threshold][budget][accumulator].values()))
                        csv_writer.writerow(result_data)


if __name__ == '__main__':
    main()
