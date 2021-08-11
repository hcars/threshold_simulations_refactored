using LightGraphs;
using GraphIO;
using Gurobi;
using Random;
using Test;
using DelimitedFiles;
using Dates;
include("./DiffusionModel.jl")
include("./Blocking.jl")
include("./SeedSelection.jl")



function main()
		# Parse CLAs
		name = ARGS[1]
		repetitions = parse(Int, ARGS[2])
		seeding_method = ARGS[3]
		num_seeds = parse(Int, ARGS[4])
		random_seed = parse(Int, ARGS[5])
		out_file_name = ARGS[6]
                println(ARGS)

		thresholds = [3]
		graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
		graph = SimpleGraph(graph_di)
		Random.seed!(random_seed)

		max_time_step = Int(nv(graph)) * 2
		epi_curve_matrix = zeros((10, max_time_step))

		for i=1:repetitions
			model = DiffusionModel.MultiDiffusionModelConstructor(graph)
			if seeding_method == "centola"
				seeds = SeedSelection.choose_by_centola(model, num_seeds)
			elseif seeding_method == "random_k_core"
				seeds = SeedSelection.choose_random_k_core(model, 20, num_seeds)
			end
			for threshold in thresholds
				for interaction_1 = 0:1
			            for interaction_2 = 0:1
					state = rand(UInt)
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
					newly_infected_nodes = zeros(max_time_step)

					updates = Vector{Tuple}()
				    updated = DiffusionModel.iterate!(model)
					# Add first set of newly infected counts.
					newly_infected_nodes[1]  = length(updated[1]) + length(updated[2])
				    max_infections = nv(model.network)
				    append!(updates, [updated])
				    iter_count = 0
				    while !(isempty(updated[1]) &&
				            isempty(updated[2]) && iter_count < max_infections)
				        updated = DiffusionModel.iterate!(model)
						# Add first set of newly infected counts.
						newly_infected_nodes[iter_count+2] = length(updated[1]) + length(updated[2])
				        if !(isempty(updated[1]) && isempty(updated[2]))
				            append!(updates, [updated])
				        end
				        iter_count += 1
				    end
                                        for j=1:max_time_step
						epi_curve_matrix[i, j] =  newly_infected_nodes[j]
					end
				end
				end
				end




		end

		writedlm(out_file_name,  epi_curve_matrix, ',')
end






main()
