from os.path import exists

import networkx as nx


def main():
    # Load in networks
    network_folder = "experiment_networks/"
    seeds = (6893, 20591, 20653)
    net_names = ["astroph", "jazz", "wiki"]
    thresholds = (2, 3, 5)
    budgets = (.1, .2, .3)
    minimum_size = lambda x: 10 if x == "jazz" else 20
    for i in range(len(net_names)):
        net_name = net_names[i]
        G = nx.read_edgelist(network_folder + net_name + '.edges', nodetype=int, create_using=nx.Graph)
        # If there is a node file add it.
        if exists(net_name + ".nodes"):
            with open(network_folder + net_name + ".nodes", 'r') as node_fp:
                node_str = node_fp.readline().strip('\n')
                node_id = int(node_str)
                G.add_node(node_id)
        # Select seed set
        k_core = nx.k_core(G, 1)
        k = 2
        # Keep increasing k core until it is of size 20 or the subgraph gets too small
        while k < 21:
            curr_core = nx.k_core(G, k)
            if curr_core.number_of_nodes() >= minimum_size(net_name):
                k_core = curr_core
            else:
                break
            k += 1
        print(list(k_core.nodes()))



if __name__ == '__main__':
    main()
