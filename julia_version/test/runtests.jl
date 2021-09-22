include("../DiffusionModel.jl")
include("../Blocking.jl")
include("../MCICH_SMC_Helpers.jl")
include("../SeedSelection.jl")
using LightGraphs;
using Random;
using Test;
using GLPK;
using GraphIO;

@testset "All Tests" begin

@testset "Propogation Test: Low Threshold" begin
    my_graph_1 = path_graph(5)
    num_contagions = 2
    node_states = zeros(UInt, (nv(my_graph_1), num_contagions) )
    node_states[1, 1] = 1
    blocked = Dict{Int, Set}()
    thresholds = fill(UInt(1), (nv(my_graph_1), num_contagions))
    interactions = zeros(UInt, num_contagions, num_contagions)
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states, thresholds, blocked, interactions, UInt32(0))
    DiffusionModel.iterate!(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 2
    DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 5
    for node in vertices(model.network)
        @test model.states[node, 1] == 1
    end
end


@testset "Propogation Test: Higher Threshold" begin
    my_graph_1 = path_graph(5)
    num_contagions = 2
    node_states = zeros(UInt, (nv(my_graph_1), num_contagions) )
    node_states[1, 1] = 1
    blocked = Dict{Int, Set}()
    thresholds = fill(UInt(2), (nv(my_graph_1), num_contagions))
    interactions = zeros(UInt, num_contagions, num_contagions)
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states, thresholds, blocked, interactions, UInt32(0))
    DiffusionModel.iterate!(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 1
    DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 1
    @test summary[1] == 4
end

@testset "Propogation Test 3" begin
    my_graph_1 = path_graph(5)
    num_contagions = 2
    node_states = zeros(UInt, (nv(my_graph_1), num_contagions) )
    node_states[1, 1] = 1
    node_states[3, 1] = 1
    node_states[5, 1] = 1
    blocked = Dict{Int, Set}()
    thresholds = fill(UInt(2), (nv(my_graph_1), num_contagions))
    interactions = zeros(UInt, num_contagions, num_contagions)
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states, thresholds, blocked, interactions, UInt32(0))
    DiffusionModel.iterate!(model)
    DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 5
    @test summary[1] == 0
    @test model.t == 2
end


@testset "Propogation Test 4" begin
    my_graph_1 = path_graph(5)
    num_contagions = 2
    node_states = zeros(UInt, (nv(my_graph_1), num_contagions) )
    node_states[1, 1] = 1
    node_states[5, 2] = 1
    blocked = Dict{Int, Set}()
    blocked_set = get!(blocked, 3, Set())
    union!(blocked_set, [2, 1])
    delete!(blocked, 3)
    get!(blocked, 3, blocked_set)
    thresholds = fill(UInt(1), (nv(my_graph_1), num_contagions))
    interactions = zeros(UInt, num_contagions, num_contagions)
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states, thresholds, blocked, interactions, UInt32(0))
    DiffusionModel.full_run(model)
    summary = DiffusionModel.getStateSummary(model)
    @test summary[2] == 2
    @test summary[1] == 1
    @test summary[3] == 2
    @test model.t == 1
end
#
#
@testset "Blocking Test: Block on Path Graph" begin
    my_graph_1 = path_graph(5)
    add_vertex!(my_graph_1)
    add_edge!(my_graph_1, 6, 3)
    num_contagions = 2
    node_states = zeros(UInt, (nv(my_graph_1), num_contagions) )
    node_states[1, 1] = 1
    node_states[6, 1] = 1
    blocked = Dict{Int, Set}()
    thresholds = fill(UInt(1), (nv(my_graph_1), num_contagions))
    interactions = zeros(UInt, num_contagions, num_contagions)
    model = DiffusionModel.MultiDiffusionModel(my_graph_1, node_states, thresholds, blocked, interactions,  UInt32(0))
    full_run_1 = DiffusionModel.full_run(model)

    @testset "High Degree Heuristic Test" begin
        blockers = Blocking.high_degree_blocking(model, [1, 0], (Set{Int}([1, 6]), Set{Int}()))
        @test length(keys(blockers)) == 1
        @test 3 in keys(blockers)
    end

    @testset "MCICH SMC Heuristic Test 1" begin
        blocker = Blocking.mcich(model, (Set{Int}(), Set{Int}()), full_run_1, [1, 2])[1]
        @test blocker[1] == Set([3])
        @test blocker[2] == Set()

        blocker = Blocking.mcich(model, (Set{Int}(), Set{Int}()), full_run_1, [1, 2], 2)[1]
        @test blocker[1] == Set([4])
        @test blocker[2] == Set()
    end

    @testset "MCICH SMC Heuristic Test 2" begin
        blocker = Blocking.mcich(model, (Set{Int}(), Set{Int}()), full_run_1, [2, 2])[1]
        @test blocker[1] == Set([2, 3])
        @test blocker[2] == Set()
    end

    @testset "MCICH SMC Heuristic Test 3" begin
        blocker = Blocking.mcich(model, (Set{Int}(), Set{Int}()), full_run_1, [2, 2], 2)[1]
        @test blocker[1] == Set([4])
        @test blocker[2] == Set()
    end

    @testset "MCICH Test Optimal ILP"  begin
        seeds = (Set{Int}([1, 6]), Set{Int}())
        blocker = Blocking.ilp_optimal(model, seeds, full_run_1, 1, GLPK.Optimizer)
        @test blocker[1] == Set([3])
        blocker = Blocking.ilp_optimal(model, seeds, full_run_1, 2, GLPK.Optimizer)
        @test blocker[1] == Set([2, 3])
        blocker = Blocking.ilp_optimal(model, seeds, full_run_1, 3, GLPK.Optimizer)
        @test blocker[1] == Set([2, 3])
    end
end


@testset "Coverage Test Set 1" begin
    @testset "Coverage Test: 1" begin
        neighbors_to_block = [Dict{Int, Set}(1=>Set([1,2,3]), 2=>Set([2,3]), 3=>Set([4]))]
        requirements = [Dict{Int, UInt}(1=>UInt(1), 2=>UInt(2), 3=>UInt(2), 4=>UInt(2))]
        budget = 2
        best_blocking, unblocked = MCICH_SMC_Helpers.compute_coverage_heuristic(neighbors_to_block, requirements, budget)
        @test best_blocking == [Set{Int}([1,2])]
        @test unblocked == 1
    end

    @testset "Coverage Test: 2" begin
        neighbors_to_block = [Dict{Int, Set}(1=>Set([1,2,3]), 2=>Set([2,3]), 3=>Set([4])), Dict{Int, Set}(1=>Set([1,2,3,4]), 2=>Set([2,3]), 3=>Set([4]))]
        requirements = [Dict{Int, UInt}(1=>UInt(1), 2=>UInt(2), 3=>UInt(2), 4=>UInt(2)), Dict{Int, UInt}(1=>UInt(1), 2=>UInt(2), 3=>UInt(2), 4=>UInt(2))]
        budget = 2
        best_blocking, unblocked = MCICH_SMC_Helpers.compute_coverage_heuristic(neighbors_to_block, requirements, budget)
        @test best_blocking == [Set{Int}([1]), Set{Int}([1])]
        @test unblocked == 6
    end

    @testset "Coverage Test: 3" begin
        neighbors_to_block = [Dict{Int, Set}(1=>Set([1,2,3]), 2=>Set([2,3]), 3=>Set([4])), Dict{Int, Set}(1=>Set([1,2,3,4]), 2=>Set([2,3]), 3=>Set([4]))]
        requirements = [Dict{Int, UInt}(1=>UInt(1), 2=>UInt(2), 3=>UInt(2), 4=>UInt(2)), Dict{Int, UInt}(1=>UInt(1), 2=>UInt(2), 3=>UInt(2), 4=>UInt(2))]
        budget = 4
        best_blocking, unblocked = MCICH_SMC_Helpers.compute_coverage_heuristic(neighbors_to_block, requirements, budget)
        @test best_blocking == [Set{Int}([1, 2]), Set{Int}([1, 2])]
        @test unblocked == 2
    end
end


@testset "Blocking Test: A simple graph that I created" begin
    # Build graph
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
    node_states_2 = zeros(UInt, 8, 2)
    node_states_2[2, 1] = 1
    node_states_2[4, 1] = 1
    node_states_2[5, 2] = 1
    node_states_2[7, 2] = 1
    node_states_copy = copy(node_states_2)

    blockedDict_2 =  Dict{Int, Set}()


    model_other = DiffusionModel.MultiDiffusionModelConstructor(my_graph_2, 2)
    model_other.states = node_states_2
    @testset "Iteration Tests" begin
        iteration_1 = DiffusionModel.iterate!(model_other)
        node_states_copy[1, 1] = 1
        node_states_copy[3, 1] = 1
        node_states_copy[6, 2] = 1
        node_states_copy[8, 2] = 1

        for i=1:size(node_states_copy)[1]
            for j=1:size(node_states_copy)[2]
                @test model_other.states[i,j] == node_states_copy[i,j]
            end
        end

        iteration_2 = DiffusionModel.iterate!(model_other)
        node_states_copy[3, 2] = 1


        for i=1:size(node_states_copy)[1]
            for j=1:size(node_states_copy)[2]
                @test model_other.states[i,j] == node_states_copy[i,j]
            end
        end
    end

    model_other = DiffusionModel.MultiDiffusionModelConstructor(my_graph_2, 2)
    node_states_2 = zeros(8, 2)
    node_states_2[2, 1] = 1
    node_states_2[4, 1] = 1
    node_states_2[5, 2] = 1
    node_states_2[7, 2] = 1
    model_other.states = node_states_2
    interactions = zeros(UInt, 2, 2)
    interactions[2, 1] = 1
    model_other.interaction_terms = interactions
    @testset "Iteration Tests 2" begin
        node_states_copy = copy(node_states_2)
        iteration_1 = DiffusionModel.iterate!(model_other)
        node_states_copy[1, 1] = 1
        node_states_copy[3, 1] = 1
        node_states_copy[6, 2] = 1
        node_states_copy[8, 2] = 1

        for i=1:size(node_states_copy)[1]
            for j=1:size(node_states_copy)[2]

                @test model_other.states[i,j] == node_states_copy[i,j]
            end
        end

        iteration_2 = DiffusionModel.iterate!(model_other)
        node_states_copy[3, 2] = 1
        node_states_copy[5, 2] = 1


        for i=1:size(node_states_copy)[1]
            for j=1:size(node_states_copy)[2]
                @test model_other.states[i,j] == node_states_copy[i,j]
            end
        end
    end

    node_states_2 = zeros(UInt, 8, 2)
    node_states_2[2, 1] = 1
    node_states_2[4, 1] = 1
    node_states_2[5, 2] = 1
    node_states_2[7, 2] = 1
    model_other = DiffusionModel.MultiDiffusionModelConstructor(my_graph_2, 2)
    model_other.states = node_states_2
    full_run_2 = DiffusionModel.full_run(model_other)


    @testset "MCICH SMC Heuristic Test" begin
        # Run and find blockers
        blocker = Blocking.mcich(model_other, (Set{Int}([2, 4]), Set{Int}([5, 7])), full_run_2, [0, 2])[1]
        @test blocker[2] == Set([6, 8])

        node_states_2 = zeros(UInt, 8, 2)
        node_states_2[2, 1] = 1
        node_states_2[4, 1] = 1
        node_states_2[5, 2] = 1
        node_states_2[7, 2] = 1
        model_other = DiffusionModel.MultiDiffusionModelConstructor(my_graph_2, 2)
        model_other.states = node_states_2
        DiffusionModel.set_blocking!(model_other, blocker)

        full_run_2 = DiffusionModel.full_run(model_other)
        @test DiffusionModel.getStateSummary(model_other)[3] == 2

    end

    @testset "MCICH ILP Test" begin
        model_other = DiffusionModel.MultiDiffusionModelConstructor(my_graph_2, 2)
        model_other.states = node_states_2
        full_run_2 = DiffusionModel.full_run(model_other)
        # Run and find optimal
        blocker = Blocking.mcich_ilp(model_other, (Set{Int}(), Set{Int}([2, 4, 5, 7])), full_run_2, [0, 1], GLPK.Optimizer)[1]
        @test blocker[2] == Set([6])
    end

    @testset "ILP Optimal Test" begin
        blocker = Blocking.ilp_optimal(model_other, (Set{Int}([2, 4]), Set{Int}([5, 7])), full_run_2, 1, GLPK.Optimizer)
        @test blocker[2] == Set([6])
        blocker = Blocking.ilp_optimal(model_other, (Set{Int}([2, 4]), Set{Int}([5, 7])), full_run_2, 2, GLPK.Optimizer)
        @test blocker[2] == Set([6, 8])
        blocker = Blocking.ilp_optimal(model_other, (Set{Int}([2, 4]), Set{Int}([5, 7])), full_run_2, 8, GLPK.Optimizer)
        @test blocker[1] == Set([1, 3])
        @test blocker[2] == Set([6, 8])

    end
end


@testset "Blocking Test: Binary Tree Graph" begin
    my_graph_3 = binary_tree(4)
    node_states_3 = zeros((nv(my_graph_3),2))
    node_states_3[1, :] .= 1

    model_3 = DiffusionModel.MultiDiffusionModelConstructor(my_graph_3, 2, [UInt(1), UInt(1)])
    model_3.states = node_states_3
    full_run_3 = DiffusionModel.full_run(model_3)
    states_3_no_block = DiffusionModel.getStateSummary(model_3)

    @testset "MCICH SMC Heuristic" begin
        blocker = Blocking.mcich(model_3, (Set{Int}([1]), Set{Int}([1])), full_run_3, [1, 1])[1]
        @test blocker[1] == Set([2]) || blocker[1] == Set([3])
        @test blocker[2] == Set([2]) || blocker[2] == Set([3])
        DiffusionModel.set_initial_conditions!(model_3, (Set{Int}([1]), Set{Int}([1])))
        DiffusionModel.set_blocking!(model_3,  blocker)
        full_run_3 = DiffusionModel.full_run(model_3)
        states_3 = DiffusionModel.getStateSummary(model_3)
        @test sum(states_3[2:3]) == sum(states_3_no_block[2:3]) - 14
    end


    @testset "MCICH ILP" begin
        blocker = Blocking.mcich_ilp(model_3, (Set{Int}([1]), Set{Int}([1])), full_run_3, [1, 1], GLPK.Optimizer)[1]
        @test blocker[1] == Set([2]) || blocker[1] == Set([3])
        @test blocker[2] == Set([2]) || blocker[2] == Set([3])
        # DiffusionModel.set_initial_conditions!(model_3, (Set{Int}([1]), Set{Int}([1])))
        # DiffusionModel.set_blocking!(model_3,  blocker)
        # DiffusionModel.full_run(model_3)
        # states_3 = DiffusionModel.getStateSummary(model_3)
        # @test sum(states_3[2:3]) + 2*states_3[4] == sum(states_3_no_block[2:3]) + 2*states_3_no_block[4] - 14
    end


    @testset "ILP Optimal Test" begin
        blocker = Blocking.ilp_optimal(model_3, (Set{Int}([1]), Set{Int}([1])), full_run_3, 4, GLPK.Optimizer)
        @test blocker[1] == Set([2, 3]) && blocker[2] == Set([2, 3])
    end


end

@testset "Blocking Test: Star Graph" begin
    my_graph_4 = star_graph(5)
    node_states_4 = zeros(UInt, 5, 2)
    node_states_4[2, :] .= 1


    model_4 = DiffusionModel.MultiDiffusionModelConstructor(my_graph_4, 2, [UInt(1), UInt(1)])
    model_4.states = node_states_4
    full_run_4 = DiffusionModel.full_run(model_4)

    @testset "MCICH SMC Heuristic" begin
        blocker = Blocking.mcich(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, [1, 1])[1]
        @test blocker[1] == Set([1])
        @test blocker[2] == Set([1])
        blocker = Blocking.mcich(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, [1, 0])[1]
        @test blocker[1] == Set([1])
        @test isempty(blocker[2])
    end

    @testset "MCICH SMC Coop Heuristic" begin
        blocker = Blocking.mcich_smc_coop(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, 2)[1]
        @test  blocker[1] == Set([1]) && blocker[2] == Set([1])
        blocker = Blocking.mcich_smc_coop(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, 1)[1]
        @test (blocker[1] == Set([1]) && blocker[2] == Set()) || (blocker[1] == Set() && blocker[2] == Set([1]))
    end


    @testset "MCICH ILP" begin
        blocker = Blocking.mcich_ilp(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, [1, 1], GLPK.Optimizer)[1]
        @test blocker[1] == Set([1])
        @test blocker[2] == Set([1])
        blocker = Blocking.mcich_ilp(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, [1, 0], GLPK.Optimizer)[1]
        @test blocker[1] == Set([1])
        @test isempty(blocker[2])
    end

    @testset "ILP Optimal Test" begin
        blocker = Blocking.ilp_optimal(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, 1, GLPK.Optimizer)
        @test (blocker[1] == Set([1]) && blocker[2] == Set([])) || (blocker[1] == Set([]) && blocker[2] == Set([1]))
        blocker = Blocking.ilp_optimal(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, 2, GLPK.Optimizer)
        @test blocker[1] == Set([1])
        @test blocker[2] == Set([1])
        blocker = Blocking.ilp_optimal(model_4, (Set{Int}([2]), Set{Int}([2])), full_run_4, 4, GLPK.Optimizer)
        @test blocker[1] == Set([1])
        @test blocker[2] == Set([1])
    end

end

@testset "Hand Drawn Graph 1" begin
    graph_5 = SimpleGraph()
    add_vertices!(graph_5, 10)
    add_edge!(graph_5, 1, 3)
    add_edge!(graph_5, 2, 3)
    add_edge!(graph_5, 2, 4)
    add_edge!(graph_5, 3, 4)
    add_edge!(graph_5, 3, 5)
    add_edge!(graph_5, 3, 6)
    add_edge!(graph_5, 4, 5)
    add_edge!(graph_5, 4, 6)
    add_edge!(graph_5, 5, 6)
    add_edge!(graph_5, 5, 8)
    add_edge!(graph_5, 6, 7)
    add_edge!(graph_5, 7, 8)
    add_edge!(graph_5, 7, 10)
    add_edge!(graph_5, 8, 9)
    add_edge!(graph_5, 9, 5)
    add_edge!(graph_5, 9, 3)
    node_states_5 = zeros((10, 2))
    node_states_5[1, 1] = 1
    node_states_5[2, 1] = 1
    node_states_5[10, 1] = 1
    node_states_5[9, 1] = 1
    blockedDict_5 = Dict{Int, Set}()
    model_5 = DiffusionModel.MultiDiffusionModelConstructor(graph_5, 2)
    model_5.states = node_states_5
    full_run_5 = DiffusionModel.full_run(model_5)
    @testset "Normal diffusion on graph 5" begin
        summary_5 = DiffusionModel.getStateSummary(model_5)
        @test summary_5[1] == 0
        @test summary_5[2] == 10
        @test summary_5[3] == 0
    end


    @testset "Optimal test on graph 5" begin
        blocker = Blocking.ilp_optimal(model_5, (Set{Int}([1,2,9,10]), Set{Int}()), full_run_5, 1, GLPK.Optimizer)
        @test blocker[1] == Set([3])
    end


    @testset "MCICH SMC test on graph 5" begin
        blocker = Blocking.mcich(model_5, (Set{Int}([1,2,9,10]), Set{Int}()), full_run_5, [1, 0])[1]
        @test blocker[1] == Set([3])
    end
end
#
# @testset "Ascending Budget" begin
#     graph_6 = binary_tree(9)
#     node_states_6 = Dict()
#     for i=1:20
#         get!(node_states_6, i, 3)
#     end
#
#     blockedDict_6 = Dict{Int,UInt}()
#     thresholdStates_6 = Dict{Int,UInt32}()
#     model_6 = DiffusionModel.MultiDiffusionModel(graph_6, node_states_6, thresholdStates_6, blockedDict_6, [UInt32(1), UInt32(1)], UInt32(0))
#     full_run_6 = DiffusionModel.full_run(model_6)
#
#     @testset "MCICH SMC test on graph 6" begin
#         summary_6 = DiffusionModel.getStateSummary(model_6)
#         active = summary_6[2] + summary_6[4]
#         for i=1:50
#             blocker = Blocking.mcich(model_6, (Set{Int}(collect(1:20)), Set{Int}(collect(1:20))), full_run_6, [i, 0])[1]
#             DiffusionModel.set_initial_conditions!(model_6, (Set{Int}(collect(1:20)), Set{Int}(collect(1:20))))
#             DiffusionModel.set_blocking!(model_6,  blocker)
#             DiffusionModel.full_run(model_6)
#             curr_sum = DiffusionModel.getStateSummary(model_6)
#             if curr_sum[2] + curr_sum[4] - 20 != 0
# 	           new_active = curr_sum[2] + curr_sum[4]
#                @test active > new_active
#                active = new_active
# 	        end
#         end
#     end
#
#
# end
#
#
# @testset "Simple Test for ILP construction" begin
#     graph_7 = SimpleGraph()
#     add_vertices!(graph_7, 4)
#     add_edge!(graph_7, 1, 2)
#     add_edge!(graph_7, 2, 3)
#     add_edge!(graph_7, 3, 4)
#
#
#     blockedDict_7 = Dict{Int,UInt}()
#     thresholdStates_7 = Dict{Int,UInt32}()
#     node_states_7 = Dict(1=>3)
#     model_7 = DiffusionModel.MultiDiffusionModel(graph_7, node_states_7, thresholdStates_7, blockedDict_7, [UInt32(1), UInt32(1)], UInt32(0))
#
#     full_run_7 = DiffusionModel.full_run(model_7)
#
#     @testset "Check ILP construction" begin
#         ilp = Blocking.ilp_construction(model_7, (Set{Int}([1]), Set{Int}([1])), full_run_7, 1, GLPK.Optimizer, collect(Int, vertices(model_7.network)))
#         x, y, z = ilp[:x_vars], ilp[:y_vars], ilp[:z_vars]
#
#
#     end
#
#
# end
#
# @testset "Simple Test for ILP construction 2" begin
#     graph_7 = SimpleGraph()
#     add_vertices!(graph_7, 6)
#     add_edge!(graph_7, 1, 4)
#     add_edge!(graph_7, 1, 5)
#     add_edge!(graph_7, 2, 4)
#     add_edge!(graph_7, 2, 5)
#     add_edge!(graph_7, 3, 4)
#     add_edge!(graph_7, 3, 5)
#     add_edge!(graph_7, 4, 6)
#     add_edge!(graph_7, 5, 6)
#
#
#     blockedDict_7 = Dict{Int,UInt}()
#     thresholdStates_7 = Dict{Int,UInt32}()
#     node_states_7 = Dict(1=>1, 2=>2, 3=>3)
#     model_7 = DiffusionModel.MultiDiffusionModel(graph_7, node_states_7, thresholdStates_7, blockedDict_7, [UInt32(2), UInt32(1)], UInt32(0))
#
#     full_run_7 = DiffusionModel.full_run(model_7)
#
#     @testset "Check ILP construction" begin
#         ilp = Blocking.ilp_construction(model_7, (Set{Int}([1, 3]), Set{Int}([2, 3])), full_run_7, 2, GLPK.Optimizer, collect(Int, vertices(model_7.network)))
#         x, y, z = ilp[:x_vars], ilp[:y_vars], ilp[:z_vars]
#         print(ilp)
#
#
#     end
#
#
# end
#
#
#
# @testset "Jazz Net Test" begin
#     name = "../../complex_net_proposal/experiment_networks/jazz.net.clean.uel"
#     graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
#     graph_8 = SimpleGraph(graph_di)
#
#     Random.seed!(129)
#
#     blockedDict_8 = Dict{Int,UInt}()
#     thresholdStates_8 = Dict{Int,UInt32}()
#     node_states_8 = Dict()
#     model_8 = DiffusionModel.MultiDiffusionModel(graph_8, node_states_8, thresholdStates_8, blockedDict_8, [UInt32(2), UInt32(2)], UInt32(0))
#     seeds = SeedSelection.choose_by_centola(model_8, 20)
#
#     DiffusionModel.set_initial_conditions!(model_8,  seeds)
#
# 	seed_set_1 = Set{Int}()
# 	seed_set_2 = Set{Int}()
# 	for node in keys(model_8.nodeStates)
# 		if model_8.nodeStates[node] == 1
# 			union!(seed_set_1, [node])
#         elseif model_8.nodeStates[node] == 2
#         	union!(seed_set_2, [node])
# 		else
# 			union!(seed_set_1, [node])
#             union!(seed_set_2, [node])
#     	end
# 	end
# 	seed_tup = (seed_set_1, seed_set_2)
#
#
#     full_run_8 = DiffusionModel.full_run(model_8)
#
#
#     @testset "MCICH SMC test on jazz" begin
#         summary_8 = DiffusionModel.getStateSummary(model_8)
#         active = summary_8[2]  + summary_8[3]  + summary_8[4]
#         previous_blockers = Dict()
#         for i=1:100
#             blocker = Blocking.mcich(model_8, seed_tup, full_run_8, [i, i])
#
#             previous_blockers = blocker
#
#             DiffusionModel.set_initial_conditions!(model_8,  seed_tup)
#             DiffusionModel.set_blocking!(model_8,  blocker[1])
#             DiffusionModel.full_run(model_8)
#             curr_sum = DiffusionModel.getStateSummary(model_8)
#             if curr_sum[2] + curr_sum[3] + curr_sum[4] - 20 != 0
#                 new_active = curr_sum[2] + curr_sum[3] + curr_sum[4]
#                 if active >= new_active
#
# 		    if !isempty(previous_blockers[1])
#                 if  length(previous_blockers[1][1]) + length(previous_blockers[1][2]) > length(blocker[1][1]) + length(blocker[1][2])
# 		    		@test blocker[2] > previous_blockers[2]
# 		        end
# 		    end
#
#                 active = new_active
#                 end
#             end
#         end
#     end
#
#
# end
#
#
end
