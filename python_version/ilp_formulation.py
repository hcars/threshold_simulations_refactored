import gurobipy as gp
import numpy as np

size_of_collection = 10
requirement_array = np.ones()

model = gp.Model('Set Multi-cover ILP')

budget = 10

node_ids = gp.multidict()  # This should be a gp.multidict from inices to a set of set indices
cost = np.ones(shape=(len(node_ids),), dtype=np.bint)
covered_elements = model.addVars(node_ids, obj=cost, name="covered elements")

coverage_array = np.ones()  # An array indicating which sets have been covered
budget_constraint = model.addConstrs(gp.quicksum(coverage_array) <= budget, name="Coverage budget.")

min_coverage = model.addConstrs(
    (sum(coverage_array[i] for i in node_ids[key]) - requirement_array[key] + 1 <=
     size_of_collection * key for key in node_ids.keys()), name="First constraint"
)

max_coverage = model.addConstrs(
    (sum(coverage_array[i] for i in node_ids[key]) - requirement_array[key] + size_of_collection >=
     size_of_collection * key for key in node_ids.keys()), name="Second constraint"
)
