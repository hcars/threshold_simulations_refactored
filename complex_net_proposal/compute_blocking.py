import copy
from os.path import exists

import networkx as nx
import numpy as np
import utils

import coverage_heuristic as cbh


def main():
    # Load in networks
    network_folder = "experiment_networks/"
    seeds = (6893, 20591, 20653)
    net_names = ["jazz", "astroph", "wiki"]
    thresholds = (2, 3, 5)
    budgets = (.1, .2, .3)
    size = lambda x: 10 if x == "jazz" else 20
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
        # Select seed set
        k_core = nx.k_core(G, 20)
        component = list(next(nx.connected_components(k_core)).nodes())
        size = size(net_name)
        seed_set_1 = []
        seed_set_2 = []
        for index in range(len(component)):
            roll = np.random.randint(1, 4)
            if roll == 3:
                seed_set_1.append(component[index])
                seed_set_2.append(component[index])
            elif roll == 2:
                seed_set_2.append(component[index])
            elif roll == 1:
                seed_set_1.append(component[index])
        # Pull out budget
        for j in range(3):
            budget = budgets[j] * G.number_of_nodes()
            # Pull out threshold
            for k in range(3):
                network = copy.deepcopy(G)
                threshold = thresholds[k]
                # Configure model
                model = utils.config_model(network, threshold, seed_set_1, seed_set_2)
                node_infections_1, node_infections_2, results = model.simulation_run()
                # Analyze node counts
                infected_1 = results['node_count'][1] + results['node_count'][3]
                infected_2 = results['node_count'][2] + results['node_count'][3]
                total_infected = sum(results['node_count'][i] for i in range(1, 4))
                # Select nodes appropriately
                if infected_1 >= infected_2:
                    ratio_total = infected_1 / total_infected
                    budget_1 = np.ceil(ratio_total * budget)
                    budget_2 = budget - budget_1
                else:
                    ratio_total = infected_2 / total_infected
                    budget_2 = np.ceil(ratio_total * budget)
                    budget_1 = budget - budget_2
                # Run through the CBH from DMKD for both contagions.
                choices_1 = cbh.try_all_sets(node_infections_1, budget_1, model, 1)
                choices_2 = cbh.try_all_sets(node_infections_2, budget_2, model, 2)
                # Run again
                network = copy.deepcopy(G)

                # Configure model
                model = utils.config_model(network, threshold, seed_set_1, seed_set_2, choices_1, choices_2)
                node_infections_1_blocked, node_infections_2_blocked, results_blocked = model.simulation_run()
                print(results_blocked, results)
                exit()


if __name__ == '__main__':
    main()
