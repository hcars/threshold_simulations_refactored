module BlockingHelpers
    using LightGraphs;


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


    function coverage_optimal(model, available_to_block::Array{Int}, to_block::Dict{Int, UInt}, budget::Int, optimizer)
        """
        This finds the optimal coverage for a set multi-cover instance.
        """
        lp = Model(optimizer)
        set_time_limit_sec(lp, 6000)
        number_sets = length(available_to_block)
        y_j = zeros(UInt, 1, number_sets)
        set_to_block = Vector{Int}(undef, length(to_block))
        i = 1
        for node in keys(to_block)
            set_to_block[i] = node
            i += 1
        end
        number_to_block = length(set_to_block)
        x_i = zeros(UInt, 1, number_to_block)
        @variable(lp, x_i[1:number_to_block])
        @variable(lp, y_j[1:number_sets])
        @constraint(lp, sum(y_j) <= budget)
        for x in x_i
            set_binary(x)
        end
        for y in y_j
            set_binary(y)
        end

        for i=1:length(set_to_block)
            node_to_block = set_to_block[i]
            neighbors_node = neighbors(model.network, node_to_block)
            k = Int(1)
            indices_neighbors = Vector{Int}()
            for j=1:length(available_to_block)
                if available_to_block[j] in neighbors_node
                    append!(indices_neighbors, [j])
                end
            end
            @constraint(lp, number_sets*x_i[i] >= sum(y_j[m] for m in indices_neighbors) - to_block[node_to_block] + 1)
            @constraint(lp, number_sets*x_i[i] <= sum(y_j[m] for m in indices_neighbors) - to_block[node_to_block] + number_sets)
        end
        @objective(lp, Max, sum(x_i))
        optimize!(lp)
        if (termination_status(lp) == MOI.TIME_LIMIT) || (termination_status(lp) == MOI.OPTIMAL)
        blockers = Set{Int}()
        for (index, y) in enumerate(y_j)
            if value.(y) == 1
                union!(blockers, [available_to_block[index]])
            end
        end
        return blockers, number_to_block - objective_value(lp)
        else
           println(termination_status(lp))
           error("The model did not solve or run out of time.")
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
                    requirements[neighbor] =  neighbors_at_infection_time - threshold
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


    function compute_candidate_blockers_mcich(model, available_to_block, next_time_step_infections, curr_contagion, curr_budget, optimizer)
        """
        This computes a set of candidate blockers for the MCICH heuristic at a given
            time step using the ILP solution.
        """
        begin

        # This computes the requirements and the set of neighbors that need to blocked for each blocking node candidate.
        neighbors_to_block, requirements = compute_requirements_mcich(model, available_to_block, next_time_step_infections, curr_contagion)
        neighbors_to_block = collect(keys(neighbors_to_block))
        # Compute the coverage with the greedy heuristic
        current_blocking, unblocked = coverage_optimal(neighbors_to_block, requirements, curr_budget, optimizer)

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

    function compute_blocking_choice_mcich(model, updates, curr_contagion, curr_budget, curr_seed_nodes, optimizer, min_time, max_time)
        """
        Given the current contagion and the other necessary information, this computes the MCICH blocking choice for the current
        contagion using the ILP for the set multi-cover problem.
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
                current_blocking, curr_unblocked = compute_candidate_blockers_mcich(model, available_to_block, next_time_step_infections, curr_contagion, curr_budget, optimizer)
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


end
