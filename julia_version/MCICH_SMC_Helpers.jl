module MCICH_SMC_Helpers
    using LightGraphs;
    using JuMP;


    function compute_maximum_intersection(neighbors_to_block, requirements, possible_blocking_nodes, num_contagions)
        """
        This computes the maximum intersection of the unblocked nodes and the available blocking nodes.
        """
        begin
            # Get the set of possible blocking nodes for each contagion.
            intersections = Array{Int}(undef, num_contagions)

            best_contagion, best_node_index, max = 1, 1, 0
            for curr_contagion=1:num_contagions
                curr_possible_blocking_nodes = possible_blocking_nodes[curr_contagion]
                # Get the current set of unblocked nodes
                curr_unblocked = keys(requirements[curr_contagion])
                # Get the intersection of each contagion with the set of unblocked nodes.
                curr_intersections = map(x->length(intersect(neighbors_to_block[curr_contagion][x], curr_unblocked)), curr_possible_blocking_nodes)
                # Compute the maximum of the intersections between the unblocked and the possible blocking nodes.
                curr_max = maximum(curr_intersections)
                if  curr_max > max
                    # Update the maximums and the indices if the maximum intersection is surpassed.
                    max = curr_max
                    best_contagion, best_node_index = curr_contagion, argmax(curr_intersections)
                end
            end
            return best_contagion, best_node_index
        end
    end





    function compute_coverage_heuristic(neighbors_to_block::Vector{Dict{Int, Set}}, requirements::Vector{Dict{Int, UInt}}, budget::Int)
        """
        Computes the set multi-cover greedy heursitic.
        """
        begin
        # Get the minimum of the budget and the possible blocking nodes.
        upperLimit = minimum([budget, sum(map(x->length(x), neighbors_to_block))])
        # Get the number of contagions
        num_contagions = length(neighbors_to_block)

        best_blocking = Vector{Set{Int}}(undef, num_contagions)
        for curr_contagion=1:num_contagions
            best_blocking[curr_contagion] = Set{Int}()
        end

        for i=1:upperLimit
            possible_blocking_nodes = map(x->collect(keys(x)), neighbors_to_block)
            # Compute the maximum intersection across available nodes and contagions.
            best_contagion, best_node_index  = compute_maximum_intersection(neighbors_to_block, requirements, possible_blocking_nodes, num_contagions)

            # Get the ID of the best node.
            best_node = possible_blocking_nodes[best_contagion][best_node_index]
            # Join the best node to blocking set.
            union!(best_blocking[best_contagion], [best_node])
            # Iterate over the nodes neigboring the node chosen.
            for node in neighbors_to_block[best_contagion][best_node]
                # If the node is in the requirements dictionary.
                curr_requirements = requirements[best_contagion]
                if node in keys(curr_requirements)
                    # Decrement its coverage requirement.
                    curr_requirements[node] -= 1
                    # If the requirement is 0, remove the node from the uncovered dictionary.
                    if curr_requirements[node] == 0
                        delete!(curr_requirements, node)
                    end
                end
            end
            # Remove the chose node from the set of nodes available to chose from.
            delete!(neighbors_to_block[best_contagion], best_node)
            # Check if all requirements are empty.
            if reduce(&, map(x->isempty(x), requirements))
                break
            end
        end
        unblocked = reduce(+, map(x->length(x), requirements))
        return best_blocking, unblocked
        end
    end




    function compute_requirements_mcich(model, available_to_block, next_time_step_infections, curr_contagion)
        """
        Computes the coverage requirement for the nodes that are in the set of nodes
        next to be infected.
        """
        begin
        requirements = Dict{Int, UInt}()
        # Get a vector that maps nodes to block to their neighbors
        neighbors_to_block = Dict{Int, Set}()
        # Iterate through the nodes that may be blocked at the current time step.
        for node in available_to_block
            curr_neighbors_to_block = Set{Int}()
            for neighbor in all_neighbors(model.network, node)
                if haskey(next_time_step_infections, neighbor)
                    # Get the number of neighbors at the time of infection.
                    neighbors_at_infection_time = next_time_step_infections[neighbor]
                    # Get the threshold for the contagion and the node
                    threshold = model.states[neighbor, curr_contagion]
                    # Add the node to the requirements dictionary
                    requirements[neighbor] =  neighbors_at_infection_time - threshold + 1
                    # Add the neighbor to the list of nodes to block for the current node
                    union!(curr_neighbors_to_block, [neighbor])
                end
            end
            # Put the node in the neighbors_to_block to block dictionary if it can block anything.
            if ! isempty(curr_neighbors_to_block)
                neighbors_to_block[node] = curr_neighbors_to_block
            end
        end
        # Return the dictionary with the possible blocking nodes and the numbers that need to be blocked
        # and return the dictionary with the coverage requirements
        return neighbors_to_block, requirements
        end
    end


    function compute_candidate_blockers_mcich(model, available_to_block, next_time_step_infections, curr_contagion, curr_budget)
        """
        This computes a set of candidate blockers for the MCICH heuristic at a given
            time step using the greedy heuristic.
        """
        begin

        # This computes the requirements and the set of neighbors that need to blocked for each blocking node candidate.
        neighbors_to_block, requirements = compute_requirements_mcich(model, available_to_block, next_time_step_infections, curr_contagion)
        # Compute the coverage with the greedy heuristic
        current_blocking, unblocked = compute_coverage_heuristic([neighbors_to_block], [requirements], curr_budget)

        return current_blocking, unblocked
        end
    end




    function compute_blocking_choice_mcich(model, updates, curr_contagion, curr_budget, curr_seed_nodes, min_time, max_time)
        """
        Given the current contagion and the other necessary information, this computes the MCICH blocking choice for the current
        contagion.
        """
        begin
            # Compute the candidate blockers
            candidate_blocker = Set{Int}()
            # Initialize the minimum number of nodes left unblocked to infinity.
            unblocked_min = Inf
            # Initialize the blocking point to be the min time.
            blocking_point = min_time
            # Computes the best blocking set for each time step.
            for curr_time_step=min_time:max_time
                # Gets updated list for the current time step
                curr_time_step_infections = updates[curr_time_step]
                # Get the updated dictionary for the current contagions
                curr_contagion_time_step_infections = curr_time_step_infections[curr_contagion]
                if isempty(curr_contagion_time_step_infections)
                    break
                end
                # Here we are differencing the seed nodes out from the set of
                # newly infected nodes for th current contagion at this time step.
                curr_contagion_time_step_infections_nodes = Set(keys(curr_contagion_time_step_infections))
                available_to_block = setdiff(curr_contagion_time_step_infections_nodes, curr_seed_nodes)
                # Check if the number of nodes to block is less than the blocking budget for the current contagion
                if length(available_to_block) <= curr_budget
                    candidate_blocker = available_to_block
                    blocking_point = curr_time_step
                    break
                end
                # Get the dictionary of newly infected nodes for the contagion at the next time step.
                next_time_step_infections = updates[curr_time_step+1][curr_contagion]
                if isempty(next_time_step_infections)
                    break
                end

                # Compute the candidate blocking set and the number of nodes unblocked for the current time step.
                current_blocking, curr_unblocked = compute_candidate_blockers_mcich(model, available_to_block, next_time_step_infections, curr_contagion, curr_budget)
                current_blocking = current_blocking[1]

                # If the current set has no unblocked nodes, we chose that one as blocking set.
                if curr_unblocked == 0
                    candidate_blocker = current_blocking
                    blocking_point = curr_time_step
                    break
                # Otherwise we update the our choice if it improves on prior blocking sets.
                elseif curr_unblocked < unblocked_min
                    candidate_blocker = current_blocking
                    unblocked_min = curr_unblocked
                    blocking_point  = curr_time_step
                end
            end

        end
        return candidate_blocker, blocking_point
    end

    function compute_requirements_mcich_coop(model, available_to_block, next_time_step_infections)
        """
        Helper function that gets the coverage requirements and the mapping between nodes and their blocking sets for the greedy SMC.
        """
        num_contagions = size(model.states)[2]
        # Initialize the requirements and neighbors to block.
        requirements = Vector{Dict{Int, UInt}}(undef, num_contagions)
        neighbors_to_block = Vector{Dict{Int, Set}}(undef, num_contagions)


        next_dict = Vector{Dict}(undef, num_contagions)
        for curr_contagion=1:num_contagions
            curr_neighbors_to_block = Dict{Int, Set}()
            curr_requirements = Dict{Int, UInt}()
            next_dict[curr_contagion] = next_time_step_infections[curr_contagion]
            for node in available_to_block[curr_contagion]
                curr_node_to_block = Set{Int}()
                # Check if the nodes in the next set of infected nodes is in the neighbor of the current set of infected nodes.
                curr_neighbors = all_neighbors(model.network, node)
                for neighbor in curr_neighbors
                    # Get the set of neighbors in the next infected set neigboring a potential blocking node.
                    # Also, find those neighbors' requirements.
                    if haskey(next_dict[curr_contagion], neighbor)
                        interaction_term = 0
                        total_interaction = 0
                        for other_contagion=1:num_contagions
                            if curr_contagion == other_contagion
                                continue
                            elseif model.states[neighbor, curr_contagion] == 1
                                total_interaction += model.interaction_terms[curr_contagion, other_contagion]
                            end
                        end

                        # Compute requirements.
                        infected_neighbors = next_dict[curr_contagion][neighbor]
                        threshold = model.thresholds[neighbor, curr_contagion]
                        requirement = infected_neighbors + total_interaction + 1 - threshold
                        # Add requirements to the current requirements dictionary.
                        curr_requirements[neighbor] = requirement
                        union!(curr_node_to_block, [neighbor])
                    end
                end
                curr_neighbors_to_block[node] = curr_node_to_block
            end
            # Add the current dictionaries to the overall ones.
            requirements[curr_contagion] = curr_requirements
            neighbors_to_block[curr_contagion] = curr_neighbors_to_block
        end

        return neighbors_to_block, requirements
    end






end
