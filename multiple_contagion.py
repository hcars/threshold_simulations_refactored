import numpy as np
from ndlib.models.DiffusionModel import DiffusionModel


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
                threshold_2 = self.params['nodes']["threshold_2"][u]
                if u_status == 0:
                    # Counts the infected status of neighbors and updates appropriately.
                    cnt_infected = 0
                    cnt_infected_2 = 0
                    for v in self.graph.neighbors(u):
                        if self.status[v] == 1:
                            cnt_infected += 1
                        elif self.status[v] == 2:
                            cnt_infected_2 += 1
                        elif self.status[v] == 3:
                            cnt_infected += 1
                            cnt_infected_2 += 1
                    satisfied_1 = threshold_1 <= cnt_infected
                    satisfied_2 = threshold_2 <= cnt_infected_2
                    if satisfied_1 and satisfied_2 and not (
                            self.params['nodes']['blocked_1'][u] or self.params['nodes']['blocked_2'][u]):
                        actual_status[u] = 3
                        if first_infected:
                            first_infected_1.add(u)
                            first_infected_2.add(u)
                            self.graph.nodes[u]['affected_1'] = cnt_infected
                            self.graph.nodes[u]['affected_2'] = cnt_infected_2
                    elif satisfied_1 and not self.params['nodes']['blocked_1'][u]:
                        actual_status[u] = 1
                        if first_infected:
                            first_infected_1.add(u)
                            self.graph.nodes[u]['affected_1'] = cnt_infected
                    elif satisfied_2 and self.params['nodes']['blocked_2'][u]:
                        actual_status[u] = 2
                        if first_infected:
                            first_infected_2.add(u)
                            self.graph.nodes[u]['affected_2'] = cnt_infected_2
                elif u_status == 1:
                    if self.params['nodes']['blocked_2'][u]:
                        continue
                    # Counts the infected status of neighbors, updates the threshold based on the interaction, and
                    # updates node state appropriately.
                    threshold_2 += int(threshold_2 * self.params['model']["interaction_1"])
                    cnt_infected_2 = 0
                    for v in self.graph.neighbors(u):
                        if self.status[v] == 2:
                            cnt_infected_2 += 1
                        elif self.status[v] == 3:
                            cnt_infected_2 += 1
                    if threshold_2 <= cnt_infected_2:
                        actual_status[u] = 3
                        if first_infected:
                            first_infected_2.add(u)
                            self.graph.nodes[u]['affected_2'] = cnt_infected_2
                elif u_status == 2:
                    if self.params['nodes']['blocked_1'][u]:
                        continue
                    # Counts the infected status of neighbors, updates the threshold based on the interaction, and
                    # updates node state appropriately.
                    threshold_1 += int(threshold_1 * self.params['model']["interaction_2"])
                    cnt_infected = 0
                    for v in self.graph.neighbors(u):
                        if self.status[v] == 1:
                            cnt_infected += 1
                        elif self.status[v] == 3:
                            cnt_infected += 1
                    if threshold_1 <= cnt_infected:
                        actual_status[u] = 3
                        if first_infected:
                            first_infected_1.add(u)
                            self.graph.nodes[u]['affected_1'] = cnt_infected

        # identify the changes w.r.t. previous iteration
        delta, node_count, status_delta = self.status_delta(actual_status)

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

    def simulation_run(self):
        """
        Runs simulation to a fixed point and returns the nodes that move states each time step.
        :return: The newly infected nodes at each time.
        """
        fixed_point = False
        results = None
        updated_node_list_1 = []
        updated_node_list_2 = []
        self.iteration()
        while not fixed_point:
            results = self.iteration(node_status=True, first_infected=True)
            fixed_point = results['first_infected_1'] == results['first_infected_2'] == set()
            updated_node_list_1.append(list(results['first_infected_1']))
            updated_node_list_2.append(list(results['first_infected_2']))
        return updated_node_list_1[:-1], updated_node_list_2[:-1], results
