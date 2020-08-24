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

            },
            "edges": {}
        }

    def iteration(self, node_status=True):

        self.clean_initial_status(self.available_statuses.values())
        actual_status = {node: nstatus for node, nstatus in self.status.items()}

        # if first iteration return the initial node status
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {"iteration": 0, "status": actual_status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy(), "fixed_point": False}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy(), "fixed_point": False}

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
                    if satisfied_1 and satisfied_2:
                        actual_status[u] = 3
                    elif satisfied_1:
                        actual_status[u] = 1
                    elif satisfied_2:
                        actual_status[u] = 2
                elif u_status == 1:
                    # Counts the infected status of neighbors, updates the threshold based on the interaction, and
                    # updates node state appropriately.
                    threshold_2 += threshold_2 * self.params['model']["interaction_1"]
                    cnt_infected_2 = 0
                    for v in self.graph.neighbors(u):
                        if self.status[v] == 2:
                            cnt_infected_2 += 1
                        elif self.status[v] == 3:
                            cnt_infected_2 += 1
                    if threshold_2 <= cnt_infected_2:
                        actual_status[u] = 3
                elif u_status == 2:
                    # Counts the infected status of neighbors, updates the threshold based on the interaction, and
                    # updates node state appropriately.
                    threshold_1 += threshold_1 * self.params['model']["interaction_2"]
                    cnt_infected = 0
                    for v in self.graph.neighbors(u):
                        if self.status[v] == 1:
                            cnt_infected += 1
                        elif self.status[v] == 3:
                            cnt_infected += 1
                    if threshold_1 <= cnt_infected:
                        actual_status[u] = 3

        # identify the changes w.r.t. previous iteration
        delta, node_count, status_delta = self.status_delta(actual_status)

        # update the actual status and iterative step
        self.status = actual_status
        self.actual_iteration += 1
        fixed_point = delta == {}
        # return the actual configuration (only nodes with status updates)
        # Returns a boolean to determine if the simulation has reached a fixed point.
        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy(), "fixed_point": fixed_point}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy(), "fixed_point": fixed_point}
