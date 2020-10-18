include("../DiffusionModel.jl")
include("../Blocking.jl")
using LightGraphs;
using Test;
using GLPK;

# Test propogation 1
my_graph_1 = path_graph(5)
node_states_1 = Dict{Int, UInt8}()
get!(node_states_1, 1, 1)
blockedDict_1 = Dict{Int, UInt8}()
thresholdStates_1 = Dict{Int, UInt32}()
model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(1), UInt32(1)], UInt32(0))
DiffusionModel.iterate!(model)
summary = DiffusionModel.getStateSummary(model)
@test summary[2] == 2
DiffusionModel.full_run(model)
summary = DiffusionModel.getStateSummary(model)
@test summary[2] == 5
for node in vertices(model.network)
    @test get(model.nodeStates, node, 0) == 1
end
# Test propogation 2
# Restate and try with higher threshold 
node_states_1 = Dict{Int, UInt8}()
get!(node_states_1, 1, 1)
blockedDict_1 = Dict{Int, UInt8}()
thresholdStates_1 = Dict{Int, UInt32}()
model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(2), UInt32(1)], UInt32(0))
full_run_1 = DiffusionModel.full_run(model)
summary = DiffusionModel.getStateSummary(model)
@test summary[2] == 1
for node in vertices(model.network)
    if node != 1
        @test get(model.nodeStates, node, 0) == 0
    end
end
# Test propogation 3
node_states_1 = Dict{Int, UInt8}()
get!(node_states_1, 1, 1)
blockedDict_1 = Dict{Int, UInt8}()
thresholdStates_1 = Dict{Int, UInt32}()
model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(1), UInt32(1)], UInt32(0))
summary = DiffusionModel.getStateSummary(model)
full_run_1 = DiffusionModel.full_run(model)
summary = DiffusionModel.getStateSummary(model)
@test summary[2] == 5
for node in vertices(model.network)
    @test get(model.nodeStates, node, 0) == 1
end
for i=1:length(full_run_1)
    updates = full_run_1[i]
    @test length(keys(updates[1])) == 1
end
# Test propogation 4
# Try blocking 
# blockers = Blocking.coverage(model, Set{Int}(keys(full_run_1[1][1])), Dict{Int, UInt}([(3, UInt(1))]), 1)
# @test blockers[1] == [2]
# # Test propogation 5
# # Try blocking 
# blockers = Blocking.coverage(model, Set{Int}(keys(full_run_1[1][1])), Dict{Int, UInt}([(3, UInt(1))]), 0)
# @test blockers[1] == []
# # Test propogation 6
# # Try blocking 
# blockers = Blocking.coverage(model,  Set{Int}(keys(full_run_1[1][1])), Dict{Int, UInt}([(3, UInt(1))]), 4)
# @test blockers[1] == [2]
# Test propogation 7 
# Try mcich
my_graph_1 = path_graph(5)
add_vertex!(my_graph_1)
add_edge!(my_graph_1, 6, 3)
node_states_1 = Dict{Int, UInt8}()
get!(node_states_1, 1, 1)
get!(node_states_1, 6, 1)
blockedDict_1 = Dict{Int, UInt8}()
thresholdStates_1 = Dict{Int, UInt32}()
model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(1), UInt32(1)], UInt32(0))
full_run_1 = DiffusionModel.full_run(model)
blocker = Blocking.mcich(model, (Set{Int}(), Set{Int}()), full_run_1, [1, 2])
@test blocker[1] == [3]
# Test propogation 8
# Different run
my_graph_2 = SimpleGraph()
add_vertices!(my_graph_2, 8)
add_edge!(my_graph_2, 1, 2)
add_edge!(my_graph_2, 1, 4)
add_edge!(my_graph_2, 2, 3)
add_edge!(my_graph_2, 4, 3)
add_edge!(my_graph_2, 3, 6)
add_edge!(my_graph_2, 3, 7)
add_edge!(my_graph_2, 7, 8)
add_edge!(my_graph_2, 7, 6)
add_edge!(my_graph_2, 6, 5)
add_edge!(my_graph_2, 5, 8)
node_states_2 = Dict{Int, UInt8}()
get!(node_states_2, 2, 1)
get!(node_states_2, 4, 1)
get!(node_states_2, 5, 2)
get!(node_states_2, 7, 2)
blockedDict_2 = Dict{Int, UInt8}()
thresholdStates_2 = Dict{Int, UInt32}()
model_other = DiffusionModel.MultiDiffusionModel(my_graph_2, node_states_2, thresholdStates_2, blockedDict_2, [UInt32(2), UInt32(2)], UInt32(0))
iteration_1 = DiffusionModel.iterate!(model_other)
@test model_other.nodeStates == Dict{Int, UInt8}([(2, 1), (3,1), (1,1), (4, 1), (5, 2), (6,2), (8,2), (7, 2)])
iteration_2 = DiffusionModel.iterate!(model_other)
@test model_other.nodeStates == Dict{Int, UInt8}([(2, 1), (3,3), (1, 1), (4, 1), (5, 2), (6,2), (7, 2), (8,2)])
# Test 9
# Run and find blockers
node_states_2 = Dict{Int, UInt8}()
get!(node_states_2, 2, 1)
get!(node_states_2, 4, 1)
get!(node_states_2, 5, 2)
get!(node_states_2, 7, 2)
model_other = DiffusionModel.MultiDiffusionModel(my_graph_2, node_states_2, thresholdStates_2, blockedDict_2, [UInt32(2), UInt32(2)], UInt32(0))
full_run_2 = DiffusionModel.full_run(model_other)
blocker = Blocking.mcich(model_other, (Set{Int}(), Set{Int}([2, 4, 5, 7])), full_run_2, [0, 1])
@test blocker[2] == [6]
# Test 10
# Run and find optimal 
blocker = Blocking.mcich_optimal(model_other, (Set{Int}(), Set{Int}([2, 4, 5, 7])), full_run_2, [0, 1], GLPK.Optimizer)
@test blocker[2] == [6]
# Test 11
graph_3 = binary_tree(4)
node_states_3 = Dict(1=>3)
blockedDict_3 = Dict{Int, UInt8}()
thresholdStates_3 = Dict{Int, UInt32}()
model_3 = DiffusionModel.MultiDiffusionModel(graph_3, node_states_3, thresholdStates_3, blockedDict_3, [UInt32(1), UInt32(1)], UInt32(0))
full_run_3 = DiffusionModel.full_run(model_3)
blocker = Blocking.mcich(model_3, (Set{Int}([1]), Set{Int}([1])), full_run_3, [1, 1])
@test blocker[1] == [2]
@test blocker[2] == [2]
DiffusionModel.set_blocking!(model_3,  blocker)
DiffusionModel.set_initial_conditions!(model_3, (Set{Int}([1]), Set{Int}([1])))
full_run_3 = DiffusionModel.full_run(model_3)
states_3 = DiffusionModel.getStateSummary(model_3)
@test isempty(intersect([4,5,8,9,10,11], keys(model_3.nodeStates)))
