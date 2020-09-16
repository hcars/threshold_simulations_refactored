import numpy as np
from ndlib.models.DiffusionModel import DiffusionModel
from cython.parallel import prange

class multiple_contagions(DiffusionModel):

    def __init__(self, graph):
        # Call the super class constructor
        super(self.__class__, self).__init__(graph)

        # Method name
        self.name = "Multiple_Contagion_Threshold"

        # Available node statuses
        self.available_statuses = {
            "Susceptible": 0,
            "Infected": 1,
            "Infected_2": 2,
            "Infected_Both": 3
        }
        # Exposed Parameters
        self.parameters = {
            "model": {
                "interaction_1": {"descr": "If a node is with infected 1, this is a float that raises or lowers"
                                           "the threshold for transition to 3 by taking the orginal thresheld t and"
                                           "setting t = t + (Interaction_1)*t.",
                                  "range": [-1, 1], "optional": False},
                "interaction_2": {
                    "descr": "If a node is with infected 2, this is a float that raises or lowers"
                             "the threshold for transition to 3 by taking the orginal thresheld t and"
                             "setting t = t + (Interaction_1)*t.",
                    "range": [-1, 1], "optional": False},

            },
            "nodes": {
                "threshold_1": {
                    "descr": "The threshold for infection with contagion 1.",
                    "range": [0, np.iinfo(np.uint32).max],
                    "optional": True
                },
                "threshold_2": {
                    "descr": "The threshold for infection with contagion 2.",
                    "range": [0, np.iinfo(np.uint32).max],
                    "optional": True
                },
                "blocked_1": {
                    "descr": "Blocked for contagion 1.",
                    "range": [False, True],
                    "optional": False,
                    "default": False
                },
                "blocked_2": {
                    "descr": "Blocked for contagion 2.",
                    "range": [False, True],
                    "optional": False,
                    "default": False
                },

            },
            "edges": {}
        }

    def get_activated_neighbors(self, u):
        """
        Find the number of neighbors infected with each contagion
        :param u: The node whose neighbors you would like to check.
        :return: An array with the counts.
        """
        infected_counts = np.zeros(shape=(2,))
        for v in self.graph.neighbors(u):
            if self.status[v] == 1:
                infected_counts[0] += 1
            elif self.status[v] == 2:
                infected_counts[1] += 1
            elif self.status[v] == 3:
                infected_counts[0] += 1
                infected_counts[1] += 1
        return infected_counts

    def iteration(self, node_status=True, first_infected=True):

        self.clean_initial_status(self.available_statuses.values())
        actual_status = {node: nstatus for node, nstatus in self.status.items()}

        # if first iteration return the initial node status
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            return_dict = {"iteration": self.actual_iteration - 1, "status": {},
                           "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            if node_status:
                return_dict['status'] = actual_status.copy()
            if first_infected:
                return_dict['first_infected_1'] = set()
                return_dict['first_infected_2'] = set()
            return return_dict
        if first_infected:
            first_infected_1 = set()
            first_infected_2 = set()
        # iteration inner loop
        for u in self.graph.nodes():
            # Evaluates nodes for possible updates
            u_status = self.status[u]
            if u_status == 3:
                continue
            else:
                # Retrieve the thresholds for both contagions.
                threshold_1 = self.params['nodes']["threshold_1"][u]
                threshold_2 = self.params['nodes']["threshold_1"][u]

                threshold_1 += int(threshold_1 * self.params['model']["interaction_1"])
                threshold_2 += int(threshold_2 * self.params['model']["interaction_2"])
                # Count nodes infected with different contagions
                cnts = self.get_activated_neighbors(u)
                satisfied_1 = threshold_1 <= cnts[0]
                satisfied_2 = threshold_2 <= cnts[1]
                # Counts the infected status of neighbors and updates appropriately.
                transition_1 = int(satisfied_1 and not self.params['nodes']['blocked_1'][u])
                transition_2 = (int(satisfied_2 and not self.params['nodes']['blocked_2'][u]) * 2)

                if u_status == 0:
                    total_satisfied = transition_1 + transition_2
                    # Set status based off sum of transition
                    actual_status[u] = total_satisfied
                    if first_infected:
                        if transition_1:
                            first_infected_1.add(u)
                            self.graph.nodes[u]['affected_1'] = cnts[0]
                        if transition_2:
                            first_infected_2.add(u)
                            self.graph.nodes[u]['affected_2'] = cnts[1]
                elif u_status == 1:
                    # Counts the infected status of neighbors, updates the threshold based on the interaction, and
                    # updates node state appropriately.
                    if transition_2 == 2:
                        actual_status[u] = 3
                        if first_infected:
                            first_infected_2.add(u)
                            self.graph.nodes[u]['affected_2'] = cnts[1]
                elif u_status == 2:
                    # Counts the infected status of neighbors, updates the threshold based on the interaction, and
                    # updates node state appropriately.
                    if transition_1 == 1:
                        actual_status[u] = 3
                        if first_infected:
                            first_infected_1.add(u)
                            self.graph.nodes[u]['affected_1'] = cnts[0]

        # identify the changes w.r.t. previous iteration
        delta, node_count, status_delta = self.status_delta(actual_status)
        # print(delta)
        # update the actual status and iterative step
        self.status = actual_status
        self.actual_iteration += 1
        # return the actual configuration (only nodes with status updates)
        # Returns a boolean to determine if the simulation has reached a fixed point.
        return_dict = {"iteration": self.actual_iteration - 1, "status": {},
                       "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        if node_status:
            return_dict['status'] = actual_status.copy()
        if first_infected:
            return_dict['first_infected_1'] = first_infected_1
            return_dict['first_infected_2'] = first_infected_2
        return return_dict

    def simulation_run(self, first_infected=True):
        """
        Runs simulation to a fixed point and returns the nodes that move states each time step.
        :return: The newly infected nodes at each time.
        """
        fixed_point = False
        results = None
        updated_node_list_1 = []
        updated_node_list_2 = []
        self.iteration()
        old_count = None
        while not fixed_point:
            results = self.iteration(node_status=True, first_infected=first_infected)
            fixed_point = results['node_count'] == old_count
            if first_infected:
                updated_node_list_1.append(list(results['first_infected_1']))
                updated_node_list_2.append(list(results['first_infected_2']))
            old_count = results['node_count']
        if first_infected:
            return updated_node_list_1[:-1], updated_node_list_2[:-1], results
        else:
            return results
