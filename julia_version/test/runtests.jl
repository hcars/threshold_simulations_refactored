include("../DiffusionModel.jl")
include("../Blocking.jl")
using LightGraphs;
using Test;

# Test propogation 1
my_graph_1 = path_graph(5)
node_states_1 = Dict{Int, UInt8}()
get!(node_states_1, 1, 1)
blockedDict_1 = Dict{Int, UInt8}()
thresholdStates_1 = Dict{Int, UInt32}()
model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states_1, thresholdStates_1, blockedDict_1, [UInt32(1), UInt32(1)], UInt32(0))
DiffusionModel.iterate(model)
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
blockers = Blocking.greedy_smc(model, Set{Int}(keys(full_run_1[1][1])), Dict{Int, UInt}([(3, UInt(1))]), 1)
@test blockers[1] == [2]
# Test propogation 5
# Try blocking 
blockers = Blocking.greedy_smc(model, Set{Int}(keys(full_run_1[1][1])), Dict{Int, UInt}([(3, UInt(1))]), 0)
@test blockers[1] == []
# Test propogation 6
# Try blocking 
blockers = Blocking.greedy_smc(model, Set{Int}(keys(full_run_1[1][1])), Dict{Int, UInt}([(3, UInt(1))]), 4)
@test blockers[1] == [2]
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
blocker = Blocking.mcich(model, Set{Int}(), full_run_1, [1, 2])
@test blocker[1] == [3]
