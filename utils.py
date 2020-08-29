import ndlib.models.ModelConfig as mc

from multiple_contagion import multiple_contagions


def config_model(G, threshold, seed_set_1, seed_set_2, seed_set_3=None, blocked_1=None, blocked_2=None):
    model = multiple_contagions(G)
    config = mc.Configuration()
    if blocked_1 is None:
        for node in G.nodes:
            config.add_node_configuration('blocked_1', node, False)
    else:
        for node in G.nodes:
            config.add_node_configuration('blocked_1', node, node in blocked_1)
    if blocked_2 is None:
        for node in G.nodes:
            config.add_node_configuration('blocked_2', node, False)
    else:
        for node in G.nodes:
            config.add_node_configuration('blocked_2', node, node in blocked_2)
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
    # Set configuration
    model.set_initial_status(config)
    return model
