import networkx as nx
import numpy as np
from multiple_contagion import MultipleContagionThreshold
import time
np.random.seed(12324)
# Network Definition
G = nx.barabasi_albert_graph(10000, 50)
model = MultipleContagionThreshold(G)

import ndlib.models.ModelConfig as mc

# Model Configuration
config = mc.Configuration()
config.add_model_parameter('interaction_1', 0)
config.add_model_parameter('interaction_2', 0)
config.add_node_set_configuration('threshold_1', {u: np.random.randint(1, 50) for u in G.nodes})
config.add_node_set_configuration('threshold_2', {u: np.random.randint(1, 50) for u in G.nodes})
config.add_node_set_configuration('blocked_1', {u: False for u in G.nodes})
config.add_node_set_configuration('blocked_2', {u: False for u in G.nodes})




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
    return seed_set_1, seed_set_2, seed_set_3

k_core = nx.k_core(G, 20)
seed_set_1, seed_set_2, seed_set_3 = choose_seed(k_core, 100)

config.add_model_initial_configuration('Infected', seed_set_1)
config.add_model_initial_configuration('Infected_2', seed_set_2)
config.add_model_initial_configuration('Infected_Both', seed_set_3)


model.set_initial_status(config)
now = time.time()
fixed_point = False
model.iteration()
old_count = None
while not fixed_point:
    iteration_results = model.iteration(node_status=True, first_infected=True)
    fixed_point = iteration_results['node_count'] == old_count
    old_count = iteration_results['node_count']
# first, second, results = model.simulation_run(True)
# print(results['status_delta'])
# print(first, second)
infected_both = []
# for u in G.nodes():
#     if model.status[u] == 3:
#         infected_both.append(u)
#
# sub = nx.subgraph(G, infected_both)
#
# print(len(list(nx.connected_components(sub))[0]))