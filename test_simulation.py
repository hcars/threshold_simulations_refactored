import networkx as nx
import numpy as np
from multiple_contagion import multiple_contagions

# Network Definition
G = nx.barabasi_albert_graph(1000, 25)
model = multiple_contagions(G)

import ndlib.models.ModelConfig as mc

# Model Configuration
config = mc.Configuration()
config.add_model_parameter('interaction_1', 0)
config.add_model_parameter('interaction_2', 0)
config.add_node_set_configuration('threshold_1', {u: np.random.randint(1, 50) for u in G.nodes})
config.add_node_set_configuration('threshold_2', {u: np.random.randint(1, 50) for u in G.nodes})

seed_set_1 = list({np.random.randint(0, 1000) for i in range(10)})
seed_set_2 = list({np.random.randint(0, 1000) for i in range(10)})

config.add_model_initial_configuration('Infected', seed_set_1)
config.add_model_initial_configuration('Infected_2', seed_set_2)

model.set_initial_status(config)

fixed_point = False
model.iteration()
while not fixed_point:
    iteration_results = model.iteration(node_status=True, first_infected=True)
    fixed_point = iteration_results['first_infected_1'] == iteration_results['first_infected_2'] == set()
