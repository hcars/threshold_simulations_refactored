using LightGraphs;
using GraphIO;
include("./DiffusionModel.jl")
include("./Blocking.jl")
# using DiffusionModel;
.



function main()
	net_names = ["networks/astroph.edges", "networks/fb-pages-politician.edges"]	
        for name in net_names
			graph_di = loadgraph(name, name, GraphIO.EdgeList.EdgeListFormat())
			graph = SimpleGraph(graph_di)
			nodeStates = Dict{Int, UInt8}()
			for i in 1:20
				get!(nodeStates, i, 1)
			end
			blockedDict = Dict{Int, UInt8}()
			thresholdStates = Dict{Int, UInt32}()
			model = DiffusionModel.MultiDiffusionModel(graph, nodeStates, thresholdStates, blockedDict, [UInt32(2), UInt(1)], UInt(0))
			results = DiffusionModel.full_run(model)
			println(DiffusionModel.getStateSummary(model))
			output = Blocking.mcich(model, Set{Int}(1:20), results, [100, 0])
			println(output)
			for node in output[1]
				get!(blockedDict, node, 1)
			end
			nodeStates = Dict{Int, UInt8}()
			for i in 1:20
				get!(nodeStates, i, 1)
			end
			model = DiffusionModel.MultiDiffusionModel(graph, nodeStates, thresholdStates, blockedDict, [UInt32(2), UInt(1)], UInt(0))
			results = DiffusionModel.full_run(model)
			println(DiffusionModel.getStateSummary(model))
			println(string("-------------"))
		end
end

main()