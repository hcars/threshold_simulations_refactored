module MCICH_ILP_Helpers
    using LightGraphs;
    using JuMP;

    function construct_coverage_ilp(model, available_to_block, requirements, budget, optimizer)
        """
        This constructs the ILP for the MCICH ILP.
            There are two constraints that enforce the following:
            1.If a node is not blocked then, the must be one less node blocked than needed to cover it.
            2.If a node is blocked, then, the number of nodes covering it must meet its requirement.
        """
        # Initialize the model.
        lp = Model(optimizer)
        # Initialize some containers and constants.
        number_sets = length(available_to_block)
        number_to_block = length(requirements)
        set_to_block = collect(keys(requirements))
        number_to_block = length(set_to_block)


        # Create the variables.
        y_j = zeros(UInt, 1, number_sets)
        x_i = zeros(UInt, 1, number_to_block)
        @variable(lp, x_i[1:number_to_block])
        @variable(lp, y_j[1:number_sets])
        # Set the variables to binary.
        for x in x_i
            set_binary(x)
        end
        for y in y_j
            set_binary(y)
        end
        # Set the constraints to ensure that the nodes will be blocked if possible.
        for node_to_block_ind=1:length(set_to_block)
            # Get the current node that needs to be blocked.
            node_to_block = set_to_block[node_to_block_ind]
            # Get the indices of its neighbors
            neighbors_node = neighbors(model.network, node_to_block)
            indices_neighbors = Vector{Int}()
            @inbounds @simd for curr_node_blocker_ind=1:length(available_to_block)
                if available_to_block[curr_node_blocker_ind] in neighbors_node
                    append!(indices_neighbors, [curr_node_blocker_ind])
                end
            end
            # Each constraint serves the purpose of ensuring the correct conditions exist for a node to be blocked or unblocked.
            # If a node is not blocked then, the must be one less node blocked than needed to cover it.
            @constraint(lp, number_sets * x_i[node_to_block_ind] >= sum(y_j[indices_neighbors]) - requirements[node_to_block] + 1)
            # If a node is blocked, then, the number of nodes covering it must meet its requirement.
            @constraint(lp, number_sets * x_i[node_to_block_ind] <= sum(y_j[indices_neighbors]) - requirements[node_to_block] + number_sets)
        end
        # Set the budget constraint.
        @constraint(lp, sum(y_j) <= budget)
        # Set the objective function.
        @objective(lp, Max, sum(x_i))

        return lp
    end

    function get_coverage_optimal_solution(lp, available_to_block)
        """
        This gets the solution that was found for the ILP and prepares it for use with other methods.
        """
        if (termination_status(lp) == MOI.TIME_LIMIT) || (termination_status(lp) == MOI.OPTIMAL)
        blockers = Set{Int}()
        for (index, y) in enumerate(lp[:y_j])
            if value.(y) == 1
                union!(blockers, [available_to_block[index]])
            end
        end
        else
           println(termination_status(lp))
           error("The model did not solve or run out of time.")
        end

        return blockers
    end

    function compute_coverage_optimal(model, available_to_block, requirements, budget::Int, optimizer, time_limit=1800)
        """
        This finds the optimal coverage for a set multi-cover instance.
        """
        available_to_block = collect(available_to_block)
        number_to_block = length(requirements)
        # Construct the ILP required.
        lp = construct_coverage_ilp(model, available_to_block, requirements, budget, optimizer)
        # Set the time limit.
        set_time_limit_sec(lp, time_limit)
        # Let the optimizer run.
        optimize!(lp)
        # Get the solution from the ILP.
        blockers = get_coverage_optimal_solution(lp, available_to_block)

        return blockers, number_to_block - objective_value(lp)

    end

    function compute_blocking_choice_mcich_ilp(model, updates, curr_contagion, curr_budget, curr_seed_nodes, optimizer, min_time, max_time)
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
                current_blocking, curr_unblocked = compute_coverage_optimal(model, available_to_block, next_time_step_infections, curr_budget, optimizer)
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
