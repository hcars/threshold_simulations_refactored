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

		thresholds = [2,3,4]
		graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
		graph = SimpleGraph(graph_di)
		Random.seed!(random_seed)
		if !isfile(out_file_name)
			blocking_methods=["no_block"]
			initialize_csv(out_file_name, blocking_methods)
		end

		max_time_step = Int(nv(graph)) * 2
		epi_curve_matrix = Matrix(0, max_time_step)

		for i=1:repetitions
			model = DiffusionModel.MultiDiffusionModelConstructor(graph)
			if seeding_method == "centola"
				seeds = SeedSelection.choose_by_centola(model, num_seeds)
			elseif seeding_method == "random_k_core"
				seeds = SeedSelection.choose_random_k_core(model, 20, num_seeds)
			end
			for threshold in thresholds
					state = rand(UInt)
					model.θ_i = [UInt(threshold), UInt(threshold)]
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
					newly_infected_nodes = Vector{Int}(undef, max_time_step)

					updates = Vector{Tuple}()
				    updated = iterate!(model)
					# Add first set of newly infected counts.
					append!(newly_infected_nodes, [length(updated[1]) + length(updated[2])])
				    max_infections = nv(model.network)
				    append!(updates, [updated])
				    iter_count = 0
				    while !(isempty(updated[1]) &&
				            isempty(updated[2]) && iter_count < max_infections)
				        updated = iterate!(model)
						# Add first set of newly infected counts.
						append!(newly_infected_nodes, [length(updated[1]) + length(updated[2])])
				        if !(isempty(updated[1]) && isempty(updated[2]))
				            append!(updates, [updated])
				        end
				        iter_count += 1
				    end

					epi_curve_matrix = cat(1, epi_curve_matrix, newly_infected_nodes)
				end




			end
		end

		writedlm(out_file_name,  epi_curve_matrix, ',')
end






main()
