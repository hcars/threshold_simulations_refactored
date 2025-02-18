module DiffusionModel


	using LightGraphs;




	# Define the DiffusionModel object.
	mutable struct MultiDiffusionModel
		"""
		DiffusionModel is the object that implements the threshold model for an
			arbitrary number of contagions.
		"""
		network::SimpleGraph
		states::Matrix{UInt}
		thresholds::Matrix{UInt}
		blocked::Dict{Int, Set}
		interaction_terms::Matrix{UInt}
		t::UInt32
	end

	# These constructors simplify creating DiffusionModel objects.
	function MultiDiffusionModelConstructor(graph, num_contagions::Number)
		"""
		This constructor creates a threshold model with the given number of
		contagions and input graph. The threshold is set uniformly to 2.
		"""
		states = zeros(UInt, (nv(graph), num_contagions))
		thresholds = fill(UInt(2), (nv(graph), num_contagions))
		blocked = Dict{Int, Set}()
		interaction_terms = zeros(UInt, num_contagions, num_contagions)
		return MultiDiffusionModel(graph, states, thresholds, blocked, interaction_terms, UInt(0))
	end

	function MultiDiffusionModelConstructor(graph, num_contagions::Number, uniform_thresholds::Vector{UInt})
		"""
		This constructor creates a threshold model with the given number of
		contagions and input graph. The threshold is set uniformly to 2.
		This constructor will fill in the threshold uniformly per contagion per node.
		"""
		states = zeros(UInt, (nv(graph), num_contagions))
		thresholds = Matrix{UInt}(undef, nv(graph), num_contagions)
		for curr_contagion=1:length(uniform_thresholds)
			curr_threshold = uniform_thresholds[curr_contagion]
			thresholds[:, curr_contagion] = fill(curr_threshold, nv(graph))
		end
		blocked = Dict{Int, Set}()
		interaction_terms = zeros(UInt, num_contagions, num_contagions)
		return MultiDiffusionModel(graph, states, thresholds, blocked, interaction_terms, UInt(0))
	end

	function set_initial_conditions!(model::MultiDiffusionModel, seeds::Set{Int}, num_contagions::UInt)
		"""
		This function resets the model to its initial conditions; all contagions are set to zero and the blocking nodes are removed.
		The input set of seed nodes is chosen to be infected with each contagion with 1/2 probability.
		"""
		states = zeros(UInt, (nv(model.network), num_contagions))
		model.blocked =  Dict{Int, Set}()
		for seed in seeds
			# Do a coin flip to determine if a node is infected with a contagion.
			for curr_contagion=1:num_contagions
				flip = rand(1:2)
				if flip == 2
					states[seed, curr_contagion] = 1
				end
			end

		end
		model.t = UInt32(0)
		model.states = states
	end


	function set_initial_conditions!(model::MultiDiffusionModel, seeds::Tuple)
		"""
		This function resets the model to its initial conditions; all contagions are set to zero and the blocking nodes are removed
		The seeds input will assumes there is a list of seed of nodes for each contagion.
		i.e. at index 1 there is a set of seed nodes for contagion 1. At index 4, for contagion 4
		, etc
		"""
		states = zeros(UInt, (nv(model.network), length(seeds)))
		model.blocked =  Dict{Int, Set}()
		for curr_contagion=1:length(seeds)
			curr_seed_set = seeds[curr_contagion]
			for seed in curr_seed_set
				states[seed, curr_contagion] = 1
			end
		end
		model.t = UInt32(0)
		model.states = states
	end

	function set_blocking!(model::MultiDiffusionModel, blockers::Vector)
		"""
		Given a list of lists that contain the blocking node for each contagion
		We set blocking to be true.
		"""
		blocked = Dict{Int, Set}()
		# Iterate through each contagion.
		for i=1:length(blockers)
			curr_set = blockers[i]
			# Iterate through each blocker for each contagion.
			for node in curr_set
				# Get the set of contagions already blocked for the node
				contagions_blocked = get(blocked, node, Set())
				# Add this contagion to the set
				contagions_blocked = union!(contagions_blocked, [i])
				# Delete the old set
				delete!(blocked, node)
				# Put in the new set
				get!(blocked, node, contagions_blocked)
			end
		end
		# Set the blocking
		model.blocked = blocked
	end

	function iterate!(model::MultiDiffusionModel)::Array
		"""
		This completes a one time step update. It only needs the model as input.
		It iterates through each node and checks to see if it meets the criteria
		to transition up in state.
		"""
		# This gives the history of updated nodes and how many neighbors caused the infection.
		# We create the list with this instead of fill because fill will create references
		# to the same dictionary.
		updated = Vector{Dict{Int, UInt32}}(undef, size(model.states)[2])
		num_contagions = size(model.states)[2]
		for curr_contagion=1:num_contagions
			updated[curr_contagion] = Dict{Int, UInt32}()
		end
		# Iterate through each vertex
		for u in vertices(model.network)
				# Get what contagions it's blocked for
				blocked_for = get(model.blocked, u, Set{}())
				@inbounds for curr_contagion=1:num_contagions
					# Check to ensure that it is not blocked for that contagion and the node is not already infected for the contagion.
					if (curr_contagion in blocked_for) || (model.states[u, curr_contagion] == 1)
						continue
					end
					# Initialize constants
					transition = false
					cnt_infected_neighbors = 0
					# Get the list of neighbors
					neighbors = all_neighbors(model.network, u)
					# Add the neighbors infected with that contagion
					cnt_infected_neighbors = sum(model.states[neighbors, curr_contagion])
					effective_threshold =  model.thresholds[u, curr_contagion]
					for other_contagion=1:num_contagions
						if other_contagion == curr_contagion
							continue
						elseif model.states[u, other_contagion] == 1
							effective_threshold -= model.interaction_terms[curr_contagion, other_contagion]
						end
					end
					# Determine if the node will transition states
					transition = (cnt_infected_neighbors >= effective_threshold)

					# If it will transition, add it to the updated set.
					if transition
						# Get the dictionary of updated nodes for the contagion.
						curr_updated = updated[curr_contagion]
						# Update the dictionary.
						curr_updated[u] = cnt_infected_neighbors
					end
				end

		end
		# Update node states
		@simd for curr_contagion=1:length(updated)
			# Get the updated set for each contagion
			curr_updated = updated[curr_contagion]
			# Set the node to infected if it was newly infected.
			model.states[collect(keys(curr_updated)), curr_contagion] .= 1
		end
		# Increment the time
		model.t += 1
		return updated
	end


	function getStateSummary(model::MultiDiffusionModel)
		"""
		This takes a DiffusionModel input, and it calculates the number of
		infections for each contagion.
		"""
		state_summary = zeros(size(model.states)[2] + 1)
		# Get the number of infections per each contagion.
		for curr_contagion=2:length(state_summary)
			state_summary[curr_contagion] = sum(model.states[:, curr_contagion - 1])
		end


		for u in vertices(model.network)
			# Count how many contagions u is infected with.
			total_infected_u = sum(model.states[u,:])
			# If u is not infected with anything, increment the counter for no infections
			if total_infected_u == 0
				state_summary[1]  += 1
			end
		end

		return state_summary
	end


	function number_updated(updated::Vector)
		"""
		Computes the number of nodes updated from the list of updates returned
		by iterate.
		"""
		updated_count = sum(map(x->length(x), updated))
		return updated_count
	end


	function full_run(model::MultiDiffusionModel)
		"""
		This runs the model until its state reaches a fixed point.
		"""
		# Prepare a vector to store the updated sets in.
		updates = Vector{}()
		# Get the maximum number of time steps the network may run for.
		max_time_steps = nv(model.network)
		# Create a variable to count the number of iterations
		iter_count = 0

		# Get the first iteration.
		updated = iterate!(model)
		# Join the updated the list of updated dictionaries to the overall list
		append!(updates, [updated])
		# Increment the iteration count
		iter_count += 1

		# Compute the number of updated nodes
		updated_count = number_updated(updated)

		while updated_count > 0 && (iter_count < max_time_steps)
			updated = iterate!(model)
			# Find the number updated.
			updated_count = number_updated(updated)
			# Add the updates list for this time step if there were any updated nodes.
			if updated_count > 0
				append!(updates, [updated])
				iter_count += 1
			else
				model.t -= 1
			end
		end
		# Return the time history.
		return updates
	end


end
