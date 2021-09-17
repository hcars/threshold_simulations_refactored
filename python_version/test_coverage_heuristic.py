import unittest

import ndlib.models.ModelConfig as mc
import networkx as nx
import numpy as np

import coverage_heuristic as cbh
import multiple_contagion


class TestMulticover(unittest.TestCase):

    def test_covering_1(self):
        unsatisfied = {1, 2, 3}
        subsets = [{1, 2}, {1}, {2, 3}, {3, 1}]
        coverage_requirement = {1: 1, 2: 2, 3: 1}
        coverage, chosen, unsat = cbh.greedy_smc(5, subsets, unsatisfied, coverage_requirement)
        universal = []
        for subset in coverage:
            universal = np.union1d(universal, list(subset))
        assert set(universal) == {1, 2, 3}
        assert coverage == [{1, 2}, {2, 3}]

    def test_covering_2(self):
        unsatisfied = {1, 2, 3}
        subsets = [{1, 2}, {1}, {2, 3}, {3, 1}]
        coverage_requirement = {1: 2, 2: 2, 3: 1}
        coverage, chosen, unsat = cbh.greedy_smc(3, subsets, unsatisfied, coverage_requirement)
        universal = []
        for subset in coverage:
            universal = np.union1d(universal, list(subset))
        assert set(universal) == {1, 2, 3}
        assert coverage == [{1, 2}, {2, 3}, {1}]

    def test_covering_3(self):
        unsatisfied = {1, 2, 3}
        coverage_requirement = {1: 1, 2: 2, 3: 1}
        subsets = [{1, 2}, {1}, {2, 3}, {3, 1}]
        coverage_requirement[1] += 1
        coverage_requirement[3] += 1
        coverage, chosen, unsat = cbh.greedy_smc(3, subsets, unsatisfied, coverage_requirement)
        universal = []
        for subset in coverage:
            universal = np.union1d(universal, list(subset))
        assert unsat != {}
        assert coverage == [{1, 2}, {2, 3}, {3, 1}]


class TestTryAll(unittest.TestCase):
    node_infections = [[1, 2, 3, 4], [5, 6, 7]]
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(1, 14)])
    G.add_edges_from([(i, j) for i in {1, 2, 3, 4} for j in {5, 6, 7}])
    G.add_edges_from([(5, 12), (5, 10), (5, 13)])
    for node in [i for i in range(1, 14)]:
        G.nodes[node]['affected_1'] = 3
    model = multiple_contagion.MultipleContagionThreshold(G)
    config = mc.Configuration()
    config.add_node_set_configuration('threshold_1', {u: 2 for u in G.nodes})
    config.add_node_set_configuration('threshold_2', {u: 2 for u in G.nodes})
    config.add_model_initial_configuration('Infected', [1, 2])
    config.add_model_parameter('interaction_1', 0)
    config.add_model_parameter('interaction_2', 0)
    config.add_node_set_configuration('blocked_1', {u: False for u in G.nodes})
    config.add_node_set_configuration('blocked_2', {u: False for u in G.nodes})
    model.set_initial_status(config)
    threshold_index = 1

    def test_try_all_1(self):
        solution = cbh.try_all_sets(self.node_infections, 4, self.model, seed_set=set(), contagion_index=1)
        assert list(solution) == [1, 2, 3, 4]

    def test_try_all_2(self):
        self.node_infections[0] = [1, 2, 3, 4, 10, 11, 12, 13]
        solution = cbh.try_all_sets(self.node_infections, 2, self.model, seed_set=set(), contagion_index=1)
        assert list(solution) == [1, 2]

    def test_try_all_3(self):
        self.node_infections[0] = [1, 2, 3, 4, 10, 11, 12, 13]
        solution = cbh.try_all_sets(self.node_infections, 1, self.model, seed_set=set(), contagion_index=1)
        assert list(solution) == [1]


class SimulationRun(unittest.TestCase):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 3), (2, 4), (2, 3)])
    model = multiple_contagion.MultipleContagionThreshold(G)
    config = mc.Configuration()
    config.add_node_set_configuration('threshold_1', {1: 1, 2: 1, 3: 2, 4: 2})
    config.add_node_set_configuration('threshold_2', {u: 2 for u in G.nodes})
    config.add_node_set_configuration('blocked_1', {u: False for u in G.nodes})
    config.add_node_set_configuration('blocked_2', {u: False for u in G.nodes})
    config.add_model_initial_configuration('Infected', [1, 2])
    config.add_model_parameter('interaction_1', 0)
    config.add_model_parameter('interaction_2', 0)
    model.set_initial_status(config)

    def test_run_simulation(self):
        results_1 = self.model.simulation_run()
        assert len(results_1[0]) == 1
        assert results_1[0][0] == {3} or results_1[0][0] == [3]
        assert results_1[1][0] == []


class CoverageHeuristic(unittest.TestCase):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 3), (2, 4), (2, 3)])
    model = multiple_contagion.MultipleContagionThreshold(G)
    config = mc.Configuration()
    config.add_node_set_configuration('threshold_1', {1: 1, 2: 1, 3: 1, 4: 2})
    config.add_node_set_configuration('threshold_2', {u: 2 for u in G.nodes})
    config.add_model_initial_configuration('Infected', [1, 2])
    config.add_model_parameter('interaction_1', 0)
    config.add_model_parameter('interaction_2', 0)
    config.add_node_set_configuration('blocked_1', {u: False for u in G.nodes})
    config.add_node_set_configuration('blocked_2', {u: False for u in G.nodes})
    model.set_initial_status(config)

    def test_heuristic(self):
        choice_1, choice_2 = cbh.coverage_heuristic(2, 1, model=self.model)
        assert (choice_1 == {3}) or (choice_1 == [3])
        assert len(choice_2) == 0


# def TestBlocking(unittest.TestCase):


if __name__ == '__main__':
    unittest.main()
