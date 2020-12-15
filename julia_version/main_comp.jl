using LightGraphs;
using GraphIO;
using Gurobi;
using Random;
using Test;
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
		budgets=append!([.0005, .001], collect(.005:.005:.2))
		graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
		graph = SimpleGraph(graph_di)
		Random.seed!(random_seed)
		if !isfile(out_file_name)
			blocking_methods=["no_block", "mcich_smc", "mcich_ilp", "ilp_opt", "random", "degree"]
			initialize_csv(out_file_name, blocking_methods)
		end
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
				no_blocking_results = DiffusionModel.full_run(model)
				no_block_summary = DiffusionModel.getStateSummary(model)

                                last_mcich = Nothing
				last_blockers = Nothing

				for budget in budgets
						Random.seed!(state)

						curr_budget = floor(nv(model.network)*budget)
						total_infected_1 = no_block_summary[2] + no_block_summary[4]
						total_infected = sum(no_block_summary[2:4])
						ratio_1 = total_infected_1 / total_infected
						budget_1 = Int(floor(curr_budget * ratio_1))
						budget_2 = Int(curr_budget - budget_1)
						selected_budgets =  [budget_1, budget_2]

						
						# Find the smart blocking method.

						#MCICH_SMC
						start = Dates.now()
						blockers_smart = Blocking.mcich(model, seed_tup, no_blocking_results, selected_budgets)
						finish = Dates.now()
						timing_mcich = finish - start

						DiffusionModel.set_initial_conditions!(model, seed_tup)
						DiffusionModel.set_blocking!(model, blockers_smart[1])

						DiffusionModel.full_run(model)
						blocking_summary_mcich_smc = DiffusionModel.getStateSummary(model)
                                                
                                                if last_mcich != Nothing && (last_mcich[1][1] > blocking_summary_mcich_smc[1])
                                                   println("Anomaly")
                                                   println("--------------") 
                                                   println("Last Inactive")
                                                   println(last_mcich[1][1]) 
                                                   println("Curr Inactive")
                                                   println(blocking_summary_mcich_smc[1])
 					           println("-------------")
          				           for h=1:2
                                                        println("Blocking Index")
 							println("--------------")   
                                                        println("Last")
 					           	println(last_mcich[2][h])
         						println("Current")
							println(blockers_smart[2][h]) 
						   end
                                                   if last_blockers != Nothing
                                                        println("Last Blockers")
							println(last_blockers)
						        println("Current Blockers")
							println(blockers_smart[1])
						   end
                                                end  	

						last_blockers = blockers_smart[1]
                                                last_mcich = (blocking_summary_mcich_smc[1], blockers_smart[2])

						#ILP_OPT
						start = Dates.now()
						blockers_smart = Blocking.ilp_optimal(model, seeds, no_blocking_results, Int64(curr_budget), Gurobi.Optimizer)
						finish = Dates.now()
						timing_ilp_opt = finish - start

						blocking = [Set{Int}(), Set{Int}()]
                                      		for key in keys(blockers_smart)
						    if blockers_smart[key] == 1
							union!(blocking[1], [key])
						    elseif blockers_smart[key] == 2
							union!(blocking[2], [key])
						    elseif blockers_smart[key] == 3
							union!(blocking[1], [key])
							union!(blocking[2], [key])
						    end
						end
						DiffusionModel.set_initial_conditions!(model, seed_tup)
						DiffusionModel.set_blocking!(model, blocking)

						DiffusionModel.full_run(model)
						blocking_summary_ilp_opt = DiffusionModel.getStateSummary(model)


						#MCICH_ILP
						start = Dates.now()
						blockers_smart = Blocking.mcich_optimal(model, seed_tup, no_blocking_results, selected_budgets, Gurobi.Optimizer)	
						finish = Dates.now()
						timing_mcich_ilp = finish - start


						DiffusionModel.set_initial_conditions!(model, seed_tup)
						DiffusionModel.set_blocking!(model, blockers_smart)

						DiffusionModel.full_run(model)
						blocking_summary_mcich_ilp = DiffusionModel.getStateSummary(model)

						blockers_random = Blocking.random_blocking(model, selected_budgets, seed_tup)	
						DiffusionModel.set_initial_conditions!(model, seed_tup)
						DiffusionModel.set_blocking!(model, blockers_random)
						DiffusionModel.full_run(model)
						blocking_summary_random = DiffusionModel.getStateSummary(model)

						blockers_degree = Blocking.high_degree_blocking(model, selected_budgets, seed_tup)	
						DiffusionModel.set_initial_conditions!(model, seed_tup)
						DiffusionModel.set_blocking!(model, blockers_degree)
						DiffusionModel.full_run(model)
						blocking_summary_degree = DiffusionModel.getStateSummary(model)

						blocking_summaries = [no_block_summary, blocking_summary_mcich_smc, blocking_summary_mcich_ilp, blocking_summary_ilp_opt, blocking_summary_random, blocking_summary_degree]
						metadata = [name, seeding_method, string(threshold), string(num_seeds), string(curr_budget), string(timing_mcich), string(timing_mcich_ilp), string(timing_ilp_opt) ]
						append_results(out_file_name, blocking_summaries, metadata)
				end
			end
		end
end


function initialize_csv(filename::String, blocking_methods)
	header = "network_name,seed_method,threshold,seed_size,budget_total,timing_mcich_smc,timing_mcich_ilp,timing_ilp_opt,"
	for i=1:length(blocking_methods)
		for j=0:3
			curr_count_name = blocking_methods[i] * '_' * string(j)
			if i < length(blocking_methods) ||  j < 3
				header *= curr_count_name * ','
			else
				header *= curr_count_name * '\n'
			end
		end
	end
	open(filename, "w") do io
		write(io, header)
	end;
end



function append_results(filename::String, summaries, metadata)
	result_string = ""
	for data in metadata
		result_string = result_string * data * ','
	end

	for i=1:length(summaries)
		curr_array = summaries[i]
		for j=1:4
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
	end;
end





main()
