using LightGraphs;
using GraphIO;
using Random;
using Test;
include("./DiffusionModel.jl")
include("./Blocking.jl")



function main()
		name = ARGS[1]
		repetitions = parse(Int, ARGS[2])
		seeding_method = ARGS[3]
		num_seeds = parse(Int, ARGS[4])
		random_seed = parse(Int, ARGS[5])
		out_file_name = ARGS[6]
		thresholds = [3,4]
		budgets=.05:.05:.5
		graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
		graph = SimpleGraph(graph_di)
		Random.seed!(random_seed)
		if !isfile(out_file_name)
			blocking_methods=["no_block", "mcich", "random", "degree"]
			initialize_csv(out_file_name, blocking_methods)
		end
		for i=1:repetitions
				model = DiffusionModel.MultiDiffusionModelConstructor(graph)
				if seeding_method == "centola"
					seeds = choose_by_centola(model, num_seeds)
				elseif seeding_method == "random_k_core"
					seeds = choose_random_k_core(model, 20, num_seeds)
				end
			for threshold in thresholds
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
				no_blocking_results = DiffusionModel.full_run(model)
				DiffusionModel.set_initial_conditions!(model, (seed_set_1, seed_set_2))
				no_blocking_results = DiffusionModel.full_run(model)
				no_block_summary = DiffusionModel.getStateSummary(model)
				for budget in budgets
						curr_budget = floor(nv(model.network)*budget)
						total_infected_1 = no_block_summary[2] + no_block_summary[4]
						total_infected = sum(no_block_summary[2:4])
						ratio_1 = total_infected_1 / total_infected
						budget_1 = Int(floor(curr_budget * ratio_1))
						budget_2 = Int(curr_budget - budget_1)
						selected_budgets =  [budget_1, budget_2]


						
						blockers_mcich = Blocking.mcich(model, (seed_set_1, seed_set_2), no_blocking_results, selected_budgets)	
						DiffusionModel.set_initial_conditions!(model, (seed_set_1, seed_set_2))
						DiffusionModel.set_blocking!(model, blockers_mcich)
						DiffusionModel.full_run(model)
						blocking_summary_mcich = DiffusionModel.getStateSummary(model)

						blockers_random = random_blocking(model, selected_budgets, (seed_set_1, seed_set_2))	
						DiffusionModel.set_initial_conditions!(model, (seed_set_1, seed_set_2))
						DiffusionModel.set_blocking!(model, blockers_random)
						DiffusionModel.full_run(model)
						blocking_summary_random = DiffusionModel.getStateSummary(model)

						blockers_degree = high_degree_blocking(model, selected_budgets, (seed_set_1, seed_set_2))	
						DiffusionModel.set_initial_conditions!(model, (seed_set_1, seed_set_2))
						DiffusionModel.set_blocking!(model, blockers_degree)
						DiffusionModel.full_run(model)
						blocking_summary_degree = DiffusionModel.getStateSummary(model)

						blocking_summaries = [no_block_summary, blocking_summary_mcich, blocking_summary_random, blocking_summary_degree]
						
						append_results(out_file_name, blocking_summaries, string(num_seeds), string(threshold), seeding_method, name, string(curr_budget))
				end
			end
		end
end


function initialize_csv(filename::String, blocking_methods::Vector{String})
	header = "network_name,seed_method,threshold,seed_size,budget_total,"
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



function append_results(filename::String, summaries::Vector, seed_size::String, threshold::String, seed_method::String, net_name::String, budget_total::String)
	result_string = net_name * ',' * seed_method * ',' * threshold * ',' * seed_size * ',' * budget_total * ','
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


function choose_by_centola(model, num_seeds::Int)
	chosen_vertex = rand(vertices(model.network))
	seeds = Set{Int}([chosen_vertex])
	while length(seeds) < num_seeds
		for neighbor in neighbors(model.network, chosen_vertex)
			union!(seeds, [neighbor])
			if length(seeds) >= num_seeds
				return seeds
			end
		end
		chosen_vertex = rand(neighbors(model.network, chosen_vertex))
	end
	return seeds
end

function choose_random_k_core(model, k::Int, num_seeds::Int)
	k_core_nodes = find_k_core(model.network, k)
	choices = Set{Int}()
	while length(choices) < num_seeds
		choice = rand(k_core_nodes)
		union!(choices, [choice])
	end
	return choices
end


function find_k_core(network, k::Int)
	k_cores = Set{Int}()
	for node in vertices(network)
		if degree(network, node) >= k
			union!(k_cores, [node])
		end
	end
	return k_cores
end

function high_degree_blocking(model, budgets::Vector{Int}, seed_set::Tuple)
	my_degree(x) = degree(model.network, x)
	high_degree_nodes = sort(vertices(model.network), by=my_degree, rev= true)
	blockers = Vector{Set{Int}}(undef, length(budgets))
	for i=1:length(blockers)
		curr_set = Set{Int}()
		curr_budget = budgets[i]
		curr_seeds = seed_set[i]
		j=1
		while length(curr_set) < curr_budget
			curr_selection = high_degree_nodes[j]
			if curr_selection ∉ curr_seeds
				union!(curr_set, [curr_selection])
			end
			j += 1
		end
		blockers[i] = curr_set
	end
	return blockers
end


function random_blocking(model, budgets::Vector{Int}, seed_set::Tuple)
	blockers = Vector{Set{Int}}(undef, length(budgets))
	for i=1:length(blockers)
		curr_set = Set{Int}()
		curr_budget = budgets[i]
		curr_seeds = seed_set[i]
		while length(curr_set) < curr_budget
			curr_selection = rand(vertices(model.network))
			if curr_selection ∉ curr_seeds
				union!(curr_set, [curr_selection])
			end
		end
		blockers[i] = curr_set
	end
	return blockers
end

main()
