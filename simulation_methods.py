import snap
import numpy as np

TRANSITION_FUNCTIONS = []


def define_independent_threshold_functions(thresholds):
    global TRANSITION_FUNCTIONS
    for i in range(len(thresholds)):
        TRANSITION_FUNCTIONS[i] = lambda x, y: y[i] >= thresholds[i] or x[i] is True


def define_competing_threshold_functions(thresholds, interaction_matrix):
    """

    :param thresholds: An array consisting of the transistion thresholds for each contagion
    :param interaction_matrix: A matrix that gives the interactions between the contagions.
    Reduces the value of the neighbors in state i by interaction_matrix[i][j]
    if the given node is infected with state j.
    :return:
    """
    global TRANSITION_FUNCTIONS
    for i in range(len(thresholds)):
        def transition(node_state, neighbor_states):
            if node_state[i] is True:
                return True
            for j in range(len(thresholds)):
                if j == i:
                    continue
                if node_state[j] is True:
                    neighbor_states[i] -= interaction_matrix[i][j]
            return neighbor_states[i] >= thresholds[i]

        TRANSITION_FUNCTIONS[i] = transition


def define_cooperating_threshold_functions(thresholds, interaction_matrix):
    """

    :param thresholds: An array consisting of the transistion thresholds for each contagion
    :param interaction_matrix: A matrix that gives the interactions between the contagions.
    Increases the value of the neighbors in state i by interaction_matrix[i][j]
    if the given node is infected with state j.
    :return:
    """
    global TRANSITION_FUNCTIONS
    for i in range(len(thresholds)):
        def transition(node_state, neighbor_states):
            if node_state[i] is True:
                return True
            for j in range(len(thresholds)):
                if j == i:
                    continue
                if node_state[j] is True:
                    neighbor_states[i] += interaction_matrix[i][j]
            return neighbor_states[i] >= thresholds[i]

        TRANSITION_FUNCTIONS[i] = transition


class SI_graph:
    def __init__(self, network=None, contagions=['contagion_1', 'contagion_2'],
                 transition_functions=None):
        self.network = network
        self.t = 0
        self.contagions = contagions
        self.transition_functions = transition_functions

    def read_edges_from_file(self, file_name, column_names, src_col=None, dst_col=None, delimiter=","):
        schema = snap.Schema()
        if src_col is None:
            src_col = column_names[0]
        if dst_col is None:
            dst_col = column_names[1]
        context = snap.TTableContext()
        for name in column_names:
            schema.Add(snap.TStrTAttrPr(name, snap.atInt))
        edge_table = snap.TTable.LoadSS(schema, file_name, context, delimiter, snap.TBool(False))
        self.network = snap.ToNetwork(snap.PNEANet, edge_table, src_col, dst_col, snap.aaFirst)
        for node in self.network.Nodes():
            for j in range(len(self.contagions)):
                self.network.AddIntAttrDatN(node, 0, self.contagions[self.contagions[j]])
        self.t = 0

    def set_node_attributes_from_ttable(self, file_name, node_col, column_names, delimiter=","):
        schema = snap.Schema()
        if node_col is None:
            node_col = column_names[0]
        context = snap.TTableContext()
        for name in column_names:
            schema.Add(snap.TStrTAttrPr(name, snap.atInt))
        node_table = snap.TTable.LoadSS(schema, file_name, context, delimiter, snap.TBool(False))
        for i in range(node_table.GetNumRows()):
            node_id = snap.GetIntVal(node_col, i)
            for name in column_names:
                if name == node_col:
                    continue
                node_attribute = snap.GetIntVal(name, i)
                if not self.network.IsNode(node_id):
                    self.network.AddNode(node_id)
                self.network.AddIntAttrDatN(node_id, node_attribute, name)

    def update_graph(self, node_id):
        if self.network is None or self.transition_functions is None:
            raise ValueError("The network or transition function have not yet been initialized.")
        node_iter = self.network.GetNI(node_id)
        node_states = np.empty(shape=(len(self.contagions),), dtype=np.bool)
        for i in range(len(self.contagions)):
            node_states[i] = self.network.GetIntAttrDatN(node_id, self.contagions[i]) == 1
        neighbor_iter = node_iter.GetOutEdges()
        neighbor_state_distribution = np.zeros(shape=(len(self.contagions),))
        for neighbor_id in neighbor_iter:
            for k in range(len(self.contagions)):
                neighbor_state = self.network.GetIntAttrDatN(neighbor_id, self.contagions[k])
                if neighbor_state == 1:
                    neighbor_state_distribution[k] += 1
        satisfied = np.empty(shape=(len(self.contagions),), dtype=np.bool)
        for i in range(len(self.contagions)):
            satisfied[i] = self.transition_functions(node_states, neighbor_state_distribution)
        return satisfied

    def apply_updates(self):
        for node in self.network.Nodes():
            node_id = node.GetId()
            satisfied = self.update_graph(node_id)
            for i in range(len(satisfied)):
                if satisfied[i]:
                    self.network.AddIntAttrDatN(node_id, 1, self.contagions[i])

    def simulate(self, time_steps):
        for i in range(time_steps):
            self.apply_updates()

    def get_state_vector(self):
        node_iter = self.network.Nodes
        state_vector = np.zeros(shape=(len(node_iter), len(self.contagions)), dtype=np.bool)
        i = 0
        for node in node_iter:
            node_id = node.GetId()
            for j in range(len(self.contagions)):
                state_vector[i][j] = np.bool(self.network.GetIntAttrDatN(node_id, self.contagions[j]))
        return state_vector

    def get_state_distrubtion(self):
        state_vector = self.get_state_vector()
        contagions = np.zeros(shape=(len(self.contagions),), dtype=np.uint)
        for i in range(state_vector.shape[0]):
            for j in range(state_vector.shape[1]):
                self.contagions[j] += state_vector[i][j]

