module DiffusionModel


	using LightGraphs;





	mutable struct MultiDiffusionModel 
		network::SimpleGraph
		nodeStates::Dict{Int,UInt8}
		thresholdStates::Dict{Int,UInt32}
		blockedDict::Dict{Int, UInt8}
		θ_i :: Vector{UInt32}
		t::UInt32
	end




	function iterate!(model::MultiDiffusionModel)
		"""
		This completes a one time step update.
		"""
		updated_1 = Dict{Int, UInt32}()	
		updated_2 = Dict{Int, UInt32}()
		for u in vertices(model.network)
			u_state = get(model.nodeStates, u, 0);
				if (u_state != 3)
					cnt_infected_1 = UInt32(0);
					cnt_infected_2 = UInt32(0);
						for v in all_neighbors(model.network, u) 	
							if (get(model.nodeStates, v, 0) == 1 || get(model.nodeStates, v, 0) == 3)
								cnt_infected_1 += 1;
							end
							if (get(model.nodeStates, v, 0) == 2 || get(model.nodeStates, v, 0) == 3)
								cnt_infected_2 += 1;
							end
						end
					transition_1 = (cnt_infected_1 >= get(model.thresholdStates, u, model.θ_i[1])) && ((get(model.blockedDict, u, 0) != 1) && (get(model.blockedDict, u, 0) != 3));
					transition_2 = (cnt_infected_2 >= get(model.thresholdStates, u, model.θ_i[2])) &&  ((get(model.blockedDict, u, 0) != 2) && (get(model.blockedDict, u, 0) != 3));
					old_state = u_state
					if ((transition_1 == true) && (old_state != 1))
						get!(updated_1, u, cnt_infected_1)
					end
					if ((transition_2 == true) && (old_state != 2))
						get!(updated_2, u, cnt_infected_2)
					end
				end	
		end
		for u in keys(updated_1)
			state = pop!(model.nodeStates, u, 0)
			state += 1
			get!(model.nodeStates, u, state)
		end
		for u in keys(updated_2)
			state = pop!(model.nodeStates, u, 0)
			state += 2
			get!(model.nodeStates, u, state)
		end
		model.t += 1
		return (updated_1, updated_2)
	end
							
							
	function getStateSummary(model::MultiDiffusionModel)
		state_summary = Array{Int}([0, 0, 0, 0])
		for nodeState in values(model.nodeStates)
			state_summary[nodeState + 1] += 1
		end
		state_summary[1] = nv(model.network) - sum((state_summary[2], state_summary[3], state_summary[4]))
		return state_summary
	end



	function full_run(model::MultiDiffusionModel)
		updates = Vector{Tuple}()
		updated = iterate!(model)
		append!(updates, [updated])
		while !(isempty(updated[1]) && isempty(updated[2]))
			updated = iterate!(model)
			if !(isempty(updated[1]) && isempty(updated[2]))
				append!(updates, [updated])
			end
		end
		return updates
	end


end