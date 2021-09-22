module Blocking
    using LightGraphs
    using JuMP
    include("./DiffusionModel.jl")
    include("./MCICH_SMC_Helpers.jl")
    include("./MCICH_ILP_Helpers.jl")
    include("./Optimal_ILP_Helpers.jl")


    function high_degree_blocking(model, budgets::Vector{Int}, seed_set::Tuple)::Dict{
        Int,
        Set,
    }
        """
        This method takes in a DiffusionModel, an array of budgets, and the seed sets.
        They return a dictionary of blockers.
        """
        my_degree(x) = degree(model.network, x)
        high_degree_nodes = sort(
            vertices(model.network),
            by = my_degree,
            rev = true,
        )
        blockers = Dict{Int,Set}()
        for curr_contagion = 1:size(model.states)[2]
                # Get the budget and seeds for the current contagion.
            curr_budget = budgets[curr_contagion]
            curr_seeds = seed_set[curr_contagion]
                # Set the number of blocking for the current contagion to 0.
            curr_blocked = 0
            j = 1
            while curr_blocked < curr_budget
                curr_selection = high_degree_nodes[j]
                if curr_selection in keys(blockers)
                    curr_blocked_contagions = blockers[curr_selection]
                else
                    curr_blocked_contagions = Set{Int}()
                end
                if !(curr_selection in curr_seeds)
                    union!(curr_blocked_contagions, [curr_contagion])
                    blockers[curr_selection] = curr_blocked_contagions
                    curr_blocked += 1
                end
                j += 1
            end
        end
        return blockers
    end


    function random_blocking(model, budgets::Vector{Int}, exclude::Tuple)::Vector{Set{Int}}
        """
        This returns a random set of blocking nodes.
        """
        blockers = Vector{Set{Int}}(undef, length(budgets))
        for i = 1:length(blockers)
            curr_set = Set{Int}()
            curr_budget = budgets[i]
            curr_seeds = exclude[i]
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

    function mcich(
        model,
        seed_sets::Tuple,
        updates::Array,
        budgets::Vector{Int},
        min_time = 1,
        max_time = Nothing,
    )
        """
        This function takes in a DiffusionModel, model, tuple containing the seed_sets, a set of budgets, the minimum blocking time,
            and the maximum blocking time.
        It returns a dictionary of blocking nodes using the greedy set multi-cover heuristic.
        ...
        # Arguments
        - `model::DiffusionModel`: This contains the DiffusionModel with the underlying network and other information.
        - `seed_sets::Tuple`: This is a n-tuple and each index contains the seed set for each contagion.
        - `updates::Array`: This is an array that contains a list of lists. Each list dictionarys with (newly infected node IDs)-> (number of nodes at the time of infection).
        - `budgets::Vector{Int}`: This is a vector containing the budgets for each contagion
        - `min_time`: This defaults to 1 and is the minimum time at which blocking nodes may be chosen.
        - `max_time`: This is the maximum time at which nodes may be chosen.
        ...
        """
        num_contagions = size(model.states)[2]
            # Initialize an object to store the blocking nodes in.
        blockings = Vector{Set}(undef, num_contagions)
            # Initialize the blocking point to 1 for each contagion.
        blocking_points = ones(num_contagions)
            # Set the maximum time to the set of nodes before the last set of infected nodeStates
            # if it is not already set.
        if max_time == Nothing
            max_time = length(updates) - 1
        end
        for curr_contagion = 1:length(budgets)
                    # Get the current budget
            curr_budget = budgets[curr_contagion]
                    # Get the current seed nodes
            curr_seed_nodes = seed_sets[curr_contagion]

            candidate_blocker, blocking_point = MCICH_SMC_Helpers.compute_blocking_choice_mcich(
                model,
                updates,
                curr_contagion,
                curr_budget,
                curr_seed_nodes,
                min_time,
                max_time,
            )
                    # Update the blocking_points and the blocking choices.
            blocking_points[curr_contagion] = blocking_point
            blockings[curr_contagion] = candidate_blocker
                    # Pass off the excess budget to the other contagion.
            if length(blockings) < curr_budget &&
               curr_contagion < length(curr_budgets)
                budgets[curr_contagion+1] += curr_budget - length(blockings)
            end
        end
        return blockings, blocking_points
    end



    function mcich_ilp(
        model,
        seed_sets::Tuple,
        updates::Vector,
        budgets::Vector{Int},
        optimizer,
        min_time = 1,
        max_time = Nothing,
    )
        """
        This function takes in a DiffusionModel, model, tuple containing the seed_sets, a set of budgets, an optimizer choice, the minimum blocking time,
            and the maximum blocking time.
        It returns a dictionary of blocking nodes. This uses the ILP formulation set of set multi-cover.
        ...
        # Arguments
        - `model::DiffusionModel`: This contains the DiffusionModel with the underlying network and other information.
        - `seed_sets::Tuple`: This is a n-tuple and each index contains the seed set for each contagion.
        - `updates::Array`: This is an array that contains a list of lists. Each list dictionarys with (newly infected node IDs)-> (number of nodes at the time of infection).
        - `budgets::Vector{Int}`: This is a vector containing the budgets for each contagion
        - `optimizer`: The optimizer is a JuMP optimizer for linear-programming problems.
        - `min_time`: This defaults to 1 and is the minimum time at which blocking nodes may be chosen.
        - `max_time`: This is the maximum time at which nodes may be chosen.
        ...
        """
        num_contagions = size(model.states)[2]
            # Initialize an object to store the blocking nodes in.
        blockings = Vector{Set}(undef, num_contagions)
            # Initialize the blocking point to 1 for each contagion.
        blocking_points = ones(num_contagions)
            # Set the maximum time to the set of nodes before the last set of infected nodeStates
            # if it is not already set.
        if max_time == Nothing
            max_time = length(updates) - 1
        end
        for curr_contagion = 1:length(budgets)
                    # Get the current budget
            curr_budget = budgets[curr_contagion]
                    # Get the current seed nodes
            curr_seed_nodes = seed_sets[curr_contagion]
            candidate_blocker, blocking_point = MCICH_ILP_Helpers.compute_blocking_choice_mcich_ilp(
                model,
                updates,
                curr_contagion,
                curr_budget,
                curr_seed_nodes,
                optimizer,
                min_time,
                max_time,
            )
                    # Update the blocking_points and the blocking choices.
            blocking_points[curr_contagion] = blocking_point
            blockings[curr_contagion] = candidate_blocker
                    # Pass off the excess budget to the other contagion.
            if length(blockings) < curr_budget &&
               curr_contagion < length(curr_budgets)
                budgets[curr_contagion+1] += curr_budget - length(blockings)
            end
        end
        return blockings, blocking_points
    end


    function ilp_optimal(
        model,
        seed_nodes::Tuple,
        updates::Vector,
        budget::Int,
        optimizer,
        time_limit = 86400,
    )
        """
        This computes the optimal blocking for the threshold model.
        """

        net_vertices = collect(Int, vertices(model.network))

            # Construct the ILP.
        lp = Optimal_ILP_Helpers.optimal_ilp_construction(
            model,
            seed_nodes,
            updates,
            budget,
            optimizer,
            net_vertices,
        )
            # Set the time limit.
        set_time_limit_sec(lp, time_limit)
            # Find the optimal solution.
        optimize!(lp)

            # Put the blockers into the desired form.
        blockers = Optimal_ILP_Helpers.optimal_ilp_solutions(
            model,
            lp,
            net_vertices,
        )

        return blockers

    end
    # This mapping will difference out the seed nodes from the possible blocking nodes.
    remove_seed(possible_blockers, seeds) = setdiff(possible_blockers, seeds)

    function mcich_smc_coop(
        model,
        seed_sets::Tuple,
        updates::Vector,
        budget::Int,
        min_time=1,
        max_time = Nothing,
    )
        """
        This is the interaction contagions version of the MCICH SMC algorithm.
        """

        num_contagions = size(model.states)[2]
        # Initialize an object to store the blocking nodes in.
        blockings = Vector{Set}(undef, num_contagions)
        # Initialize the blocking point to 1 for each contagion.
        blocking_points = ones(num_contagions)
        # Set the maximum time to the set of nodes before the last set of infected nodeStates
        # if it is not already set.
        if max_time == Nothing
            max_time = length(updates) - 1
        end
        blocking_point = 1
        unblocked_min = Inf
        for j = min_time:max_time
            # Get the updated nodes for the time step j.
            curr_time_step_infections = updates[j]
            find_blockings = Vector{Set}(undef, num_contagions)
            for curr_contagion=1:num_contagions
                find_blockings[curr_contagion] = Set(keys(curr_time_step_infections[curr_contagion]))
            end
            # Check if all the node sets to choose to block are empty.
            if reduce(&, map(x->isempty(x), find_blockings))
                break
            end
            # Here we are differencing the seed nodes out from the seed sets.
            # We use a mapping defined above.
            available_to_block = map(remove_seed, find_blockings, seed_sets)


            # If there are less nodes to block than the budget use available_to_block as the blocking set.
            if sum(map(x->length(x), available_to_block)) <= budget
                blockings = available_to_block
                blocking_point = j
                break
            end

            # Get the next of set of nodes infected.
            next_time_step_infections = updates[j+1]
            # Compute the necessary inputs to the coverage algorithm.
            neighbors_to_block, requirements = MCICH_SMC_Helpers.compute_requirements_mcich_coop(model, available_to_block, next_time_step_infections)
            # See if there are any nodes to block.
            if reduce(&, map(x->isempty(x), requirements))
                break
            end
            # Compute the blocking with the SMC algorithm.
            current_blocking, unblocked = MCICH_SMC_Helpers.compute_coverage_heuristic(neighbors_to_block, requirements, budget)
            if unblocked == 0
                blockings = current_blocking
                blocking_point = j
                break
            elseif unblocked < unblocked_min
                blockings = current_blocking
                unblocked_min = unblocked
                blocking_point = j
            end
        end
        return blockings, blocking_point
    end


end
