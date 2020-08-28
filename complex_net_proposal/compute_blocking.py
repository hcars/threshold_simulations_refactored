from os.path import exists
import numpy as np
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
        smallest_size = np.iinfo(np.int).max
        core_choice = None
        for component in nx.connected_components(k_core):
            component = list(k_core.nodes())
            if len(component) >= minimum_size(net_name):
               if len(component) < smallest_size:
                  core_choice = component
                  smallest_size = len(component)
        # If the component is really large, shrink it down.
        if smallest_size >= 2*minimum_size(net_name):
           for i in range(int(.5*smallest_size)):
               core_choice.pop(np.random.randint(0, len(core_choice)))             
        # Pull out budget
        for j in range(3):
        
    


if __name__ == '__main__':
    main()
