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
		max_time_step = Int(nv(graph))
		interaction_1_max = 0
		interaction_2_max = 7
		epi_curve_matrix_1 = zeros((repetitions, interaction_1_max + 1, interaction_2_max + 1, max_time_step))
		epi_curve_matrix_2 = zeros((repetitions, interaction_1_max + 1, interaction_2_max + 1, max_time_step))

		for i=1:repetitions
			model = DiffusionModel.MultiDiffusionModelConstructor(graph)


			high_core = SeedSelection.find_k_core(model.network, 30)
			low_core = SeedSelection.find_k_core(model.network, 20)

			seed_set_1 = Set{Int}()
                        seed_set_2 = Set{Int}()
			while (length(seed_set_1) < Int(floor(( num_seeds) / 5)))
			    choice = rand(high_core)
			    union!(seed_set_1, [choice])
                            low_core = setdiff(low_core, [choice])
			end
			while (length(seed_set_2) < Int(floor( (4 * num_seeds) / 5)))
			    choice = rand(low_core)
			    union!(seed_set_2, [choice])
			end
			seed_tup = (seed_set_1, seed_set_2)

			for threshold in thresholds
				for interaction_1 = 0:interaction_1_max
				        for interaction_2 = 0:interaction_2_max
						state = rand(UInt)
						model.θ_i = [UInt(3), UInt(8)]
	                    model.ξ_i = [UInt8(0), UInt8(interaction_2)]




						DiffusionModel.set_initial_conditions!(model, seed_tup)

						println(string(interaction_1) * '_' * string(interaction_2))
						println(DiffusionModel.getStateSummary(model))

						newly_infected_nodes = zeros(max_time_step)

						updates = Vector{Tuple}()
					    updated = DiffusionModel.iterate!(model)
				   	    # Add first set of newly infected counts.
					    epi_curve_matrix_1[i, interaction_1 + 1, interaction_2 + 1, 1] = length(updated[1])
					    epi_curve_matrix_2[i, interaction_1 + 1, interaction_2 + 1, 2] = length(updated[2])

					    max_infections = nv(model.network)
					    append!(updates, [updated])
					    iter_count = 2
					    while (!(isempty(updated[1]) && isempty(updated[2])) && (iter_count < max_infections))
					        updated = DiffusionModel.iterate!(model)
							# Add first set of newly infected counts.
							epi_curve_matrix_1[i, interaction_1 + 1, interaction_2 + 1, iter_count] = length(updated[1])
							epi_curve_matrix_2[i, interaction_1 + 1, interaction_2 + 1, iter_count] = length(updated[2])


					        iter_count += 1
					        if !(isempty(updated[1]) && isempty(updated[2]))
					            append!(updates, [updated])
					        end
					    end
						println(DiffusionModel.getStateSummary(model))
			


	               					end

				end
			end
			for interaction_1 = 0:interaction_1_max
				        for interaction_2 = 0:interaction_2_max

		 	writedlm(out_file_name * "_contagion_1_interaction_" * string(interaction_1) * '_' * string(interaction_2),   epi_curve_matrix_1[:, interaction_1 + 1, interaction_2 + 1, :] , ',')
	                writedlm(out_file_name * "_contagion_2_interaction_" * string(interaction_1) * '_' * string(interaction_2),  epi_curve_matrix_2[:, interaction_1 + 1, 
interaction_2 + 1, :] , ',')
			end
			end




		end

end






main()
