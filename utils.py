import ndlib.models.ModelConfig as mc
import numpy as np
from multiple_contagion import MultipleContagionThreshold


def config_model(G, threshold, seed_set_1, seed_set_2, seed_set_3=None, blocked_1=[], blocked_2=[]):
    G.remove_nodes_from(blocked_1)
    G.remove_nodes_from(blocked_2)
    model = MultipleContagionThreshold(G)
    config = mc.Configuration()
    # No interaction
    config.add_model_parameter('interaction_1', 0)
    config.add_model_parameter('interaction_2', 0)
    # Set threshold
    config.add_node_set_configuration('threshold_1', {u: threshold for u in G.nodes})
    config.add_node_set_configuration('threshold_2', {u: threshold for u in G.nodes})
    # Initialize seed set
    config.add_model_initial_configuration('Infected', seed_set_1)
    config.add_model_initial_configuration('Infected_2', seed_set_2)
    config.add_model_initial_configuration('Infected_Both', seed_set_3)
    # Set nodes to being blocked
    blocked_both = np.intersect1d(blocked_1, blocked_2)
    blocked_1 = np.setdiff1d(blocked_both, blocked_1)
    blocked_2 = np.setdiff1d(blocked_both, blocked_2)
    config.add_model_initial_configuration('Blocked', blocked_1)
    config.add_model_initial_configuration('Blocked_2', blocked_2)
    config.add_model_initial_configuration('Blocked_Both', blocked_both)
    # Set configuration
    model.set_initial_status(config)
    return model
