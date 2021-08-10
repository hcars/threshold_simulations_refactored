using LightGraphs;
using GraphIO;
using Gurobi;
using Random;
using Test;
include("./DiffusionModel.jl")
include("./Blocking.jl")
include("./SeedSelection.jl")



function main()

    print(ARGS)
	# Parse CLAs
    name = ARGS[1]
    repetitions = parse(Int, ARGS[2])
    seeding_method = ARGS[3]
    num_seeds = parse(Int, ARGS[4])
    random_seed = parse(Int, ARGS[5])
    out_file_name = ARGS[6]
    blocking_method = ARGS[7]

    thresholds = [2, 3, 4]
    budgets = append!([.0005, .001], collect(.005:.005:.07))
    graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
    graph = SimpleGraph(graph_di)
    Random.seed!(random_seed)
    if !isfile(out_file_name)
        blocking_methods = ["no_block", "mcich", "degree"]
        initialize_csv(out_file_name, blocking_methods)
    end
    for i = 1:repetitions
        model = DiffusionModel.MultiDiffusionModelConstructor(graph)
        if seeding_method == "centola"
            seeds = SeedSelection.choose_by_centola(model, num_seeds)
        elseif seeding_method == "random_k_core"
            seeds = SeedSelection.choose_random_k_core(model, 20, num_seeds)
        end
        for threshold in thresholds
            state = rand(UInt)
            model.θ_i = [UInt(threshold), UInt(threshold)]
			model.ξ_i = [UInt8(1), UInt8(0)]
            DiffusionModel.set_initial_conditions!(model, seeds)

            seed_set_1 = Set{Int}()
            seed_set_2 = Set{Int}()
            for node in keys(model.nodeStates)
                if model.nodeStates[node] == 1
                    union!(seed_set_1, [node])
                elseif model.nodeStates[node] == 2
                    union!(seed_set_2, [node])
                else
                    union!(seed_set_1, [node])
                    union!(seed_set_2, [node])
                end
            end
            seed_tup = (seed_set_1, seed_set_2)

            no_blocking_results = DiffusionModel.full_run(model)
            DiffusionModel.set_initial_conditions!(model, seed_tup)
            no_blocking_results = DiffusionModel.full_run(model)
            no_block_summary = DiffusionModel.getStateSummary(model)
            for budget in budgets
                Random.seed!(state)



				# Find the smart blocking method.
                blockers_smart = Blocking.mcich(
                    model,
                    seed_tup,
                    no_blocking_results,
                    selected_budgets,
                )[0]

                DiffusionModel.set_initial_conditions!(model, seed_tup)
                DiffusionModel.set_blocking!(model, blockers_smart)


                DiffusionModel.full_run(model)
                blocking_summary_mcich = DiffusionModel.getStateSummary(model)



                blockers_degree = Blocking.high_degree_blocking(
                    model,
                    selected_budgets,
                    seed_tup,
                )
                DiffusionModel.set_initial_conditions!(model, seed_tup)
                DiffusionModel.set_blocking!(model, blockers_degree)
                DiffusionModel.full_run(model)
                blocking_summary_degree = DiffusionModel.getStateSummary(model)

                blocking_summaries = [
                    no_block_summary,
                    blocking_summary_mcich,
                    blocking_summary_degree,
                ]
                metadata = [
                    name,
                    seeding_method,
                    string(threshold),
                    string(num_seeds),
                    string(curr_budget),
                    string(blocking_method),
                ]
                append_results(out_file_name, blocking_summaries, metadata)
                println("Append complete")
            end
        end
    end
end


function initialize_csv(filename::String, blocking_methods)
    header = "network_name,seed_method,threshold,seed_size,budget_total,smart_method,"
    for i = 1:length(blocking_methods)
        for j = 0:3
            curr_count_name = blocking_methods[i] * '_' * string(j)
            if i < length(blocking_methods) || j < 3
                header *= curr_count_name * ','
            else
                header *= curr_count_name * '\n'
            end
        end
    end
    open(filename, "w") do io
        write(io, header)
    end
end



function append_results(filename::String, summaries, metadata)
    result_string = ""
    for data in metadata
        result_string = result_string * data * ','
    end
    for i = 1:length(summaries)
        curr_array = summaries[i]
        for j = 1:4
            result_string *= string(curr_array[j])
            if i < length(summaries) || j < 4
                result_string *= ','
            else
                result_string *= '\n'
            end
        end
    end
    open(filename, "a") do io
        write(io, result_string)
    end
end


main()
