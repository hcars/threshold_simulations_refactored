using LightGraphs;
using GraphIO;
using GLPK;
include("./DiffusionModel.jl")
include("./Blocking.jl")



function main()
		name = ARGS[1]
		repetitions = Int(ARGS[2])
		graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
		graph = SimpleGraph(graph_di)
		for i=1:repetitions
			if ARGS[3] == "centola"
				seeds = choose_by_centola(model, Int(ARGS[4]))
			end
			model = DiffusionModel.MultiDiffusionModel([UInt32(2), UInt(1)])
			set_initial_conditions!(model, seeds)
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
			results = DiffusionModel.full_run(model)
			println(DiffusionModel.getStateSummary(model))
			output = Blocking.mcich(model, (seed_set_1, seed_set_2), results, [10, 0])
			println(output)
			for set in output
				for node in set
					get!(blockedDict, node, 1)
				end
			end
			model.blockedDict = blockedDict
			DiffusionModel.set_initial_conditions!(model, (seed_set_1, seed_set_2))
			results = DiffusionModel.full_run(model)
			println(DiffusionModel.getStateSummary(model))
			println(string("-------------"))
		end
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



main()