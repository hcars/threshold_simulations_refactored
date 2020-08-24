import networkx as nx
import numpy as np
from multiple_contagion import multiple_contagions

# Network Definition
G = nx.barabasi_albert_graph(1000, 25)
print(len(list(G.edges)))
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


# Simulation
iterations = model.iteration_bunch(100)
trends = model.build_trends(iterations)


from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend

viz = DiffusionTrend(model, trends)
p = viz.plot(width=1000, height=800)
show(p)
