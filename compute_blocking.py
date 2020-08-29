import copy
import pickle
from os.path import exists
import csv
import networkx as nx
import numpy as np
import utils
from multiple_contagion import multiple_contagions
import coverage_heuristic as cbh


def main():
    field_names = ['network_name', 'threshold', 'budget_total', 'budget_1', 'budget_2'] + [str(i)+"_no_block" for i in range(4)] + [str(i)+"_cbh" for i in range(4)] + [str(i)+"_degree" for i in range(4)]
    with open('complex_net_proposal/experiment_results/results.csv', 'w', newline='') as csv_fp:
         csv_writer = csv.writer(csv_fp, delimiter=',')
         csv_writer.writerow(field_names)
    # Load in networks
    network_folder = "complex_net_proposal/experiment_networks/"
    seeds = (6893, 20591, 20653)
    net_names = ["jazz", "astroph", "wiki"]
    thresholds = (2, 3, 5)
    budgets = (.1, .2, .3, .35)
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
        # Select seed set
        k_core = nx.k_core(G, 20)
        component = list(next(nx.connected_components(k_core)))
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
        # Pull out budget
        for j in range(4):
            budget = int(budgets[j] * G.number_of_nodes())
            # Pull out threshold
            for k in range(3):
                network = copy.deepcopy(G)
                threshold = thresholds[k]
                # Configure model
                model = utils.config_model(network, threshold, seed_set_1, seed_set_2, seed_set_3)
                node_infections_1, node_infections_2, results = model.simulation_run()
                # Analyze node counts
                infected_1 = results['node_count'][1] + results['node_count'][3]
                infected_2 = results['node_count'][2] + results['node_count'][3]
                total_infected = sum(results['node_count'][i] for i in range(1, 4))
                # Select nodes appropriately
                if infected_1 > infected_2:
                    ratio_total = infected_1 / total_infected
                    budget_1 = np.ceil(ratio_total * budget)
                    budget_2 = budget - budget_1
                elif infected_1 < infected_2:
                    ratio_total = infected_2 / total_infected
                    budget_2 = np.ceil(ratio_total * budget)
                    budget_1 = budget - budget_2
                else:
                    budget_1 = budget // 2
                    budget_2 = budget - budget_1
                # Run through the CBH from DMKD for both contagions.
                choices_1 = cbh.try_all_sets(node_infections_1, budget_1, model, 1)
                choices_2 = cbh.try_all_sets(node_infections_2, budget_2, model, 2)
                # Run again
                network = copy.deepcopy(G)

                # Configure model
                model = utils.config_model(network, threshold, seed_set_1, seed_set_2, seed_set_3, choices_1, choices_2)
                node_infections_1_blocked, node_infections_2_blocked, results_blocked = model.simulation_run()
                # Find high degree nodes
                network = copy.deepcopy(G)
                nodes_by_degree = sorted(G.degree(), key=lambda x: x[1])
                choices_1 = list(map(lambda x: x[0], nodes_by_degree[:budget_1]))
                choices_2 = list(map(lambda x: x[0], nodes_by_degree[:budget_2]))
                # Run forward
                model = utils.config_model(network, threshold, seed_set_1, seed_set_2, seed_set_3, choices_1, choices_2)
                node_infections_1_blocked_degree, node_infections_2_blocked_degree, results_blocked_degree = model.simulation_run()
                with open('complex_net_proposal/experiment_results/results.csv', 'a', newline='') as results_fp:
                     csv_writer = csv.writer(results_fp, delimiter=',')
                     blocked_counts = results_blocked['node_count']
                     result_data = [net_name, str(threshold), str(budget), str(budget_1), str(budget_2)] + list(map(lambda x: str(x), results['node_count'].values())) + list(map(lambda x: str(x), results_blocked['node_count'].values())) + list(map(lambda x: str(x), results_blocked_degree['node_count'].values()))
                     csv_writer.writerow(result_data)

if __name__ == '__main__':
    main()
