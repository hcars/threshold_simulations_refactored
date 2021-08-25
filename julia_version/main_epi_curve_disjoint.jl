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

		thresholds = [5]
		graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
		graph = SimpleGraph(graph_di)

		Random.seed!(random_seed)
		max_time_step = Int(nv(graph)) * 2
		epi_curve_matrix = zeros((repetitions, max_time_step))

		for i=1:repetitions
			model = DiffusionModel.MultiDiffusionModelConstructor(graph)


			high_core = SeedSelection.choose_random_k_core(model, 20, num_seeds // 2)
			low_core = SeedSelection.choose_random_k_core(model, 10, num_seeds // 2)

			seed_set_1 = high_core
			seed_set_2 = low_core


			for threshold in thresholds
				for interaction_1 = 0:3
			        for interaction_2 = 0:3
						state = rand(UInt)
						model.θ_i = [UInt(3), UInt(5)]
	                    model.ξ_i = [UInt8(interaction_1), UInt8(interaction_2)]




						DiffusionModel.set_initial_conditions!(model, seed_tup)

						println(string(interaction_1) * '_' * string(interaction_2))
						println(DiffusionModel.getStateSummary(model))

						newly_infected_nodes = zeros(max_time_step)

						updates = Vector{Tuple}()
					    updated = DiffusionModel.iterate!(model)
						# Add first set of newly infected counts.
						newly_infected_nodes[1]  = length(updated[1]) + length(updated[2])
					    max_infections = nv(model.network)
					    append!(updates, [updated])
					    iter_count = 2
					    while (!(isempty(updated[1]) && isempty(updated[2])) && (iter_count < max_infections))
					        updated = DiffusionModel.iterate!(model)
							# Add first set of newly infected counts.
							newly_infected_nodes[iter_count] = length(updated[1]) + length(updated[2])
					        if !(isempty(updated[1]) && isempty(updated[2]))
					            append!(updates, [updated])
					        end
					        iter_count += 1
					    end
			                    for j=1:max_time_step
							epi_curve_matrix[i, j] =  newly_infected_nodes[j]
					    end
						println(DiffusionModel.getStateSummary(model))


	                	writedlm(out_file_name * '_' * string(interaction_1) * '_' * string(interaction_2) ,  epi_curve_matrix , ',')
					end

				end
			end




		end

end






main()
