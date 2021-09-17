module Blocking
    using LightGraphs;
    using JuMP;
    include("./DiffusionModel.jl")
    include("./Blocking_Helpers.jl")


    function high_degree_blocking(model, budgets::Vector{Int}, seed_set::Tuple)::Dict{Int, Set}
        """
        This method takes in a DiffusionModel, an array of budgets, and the seed sets.
        They return a dictionary of blockers.
        """
        my_degree(x) = degree(model.network, x)
        high_degree_nodes = sort(vertices(model.network), by=my_degree, rev=true)
        blockers = Dict{Int, Set}()
        for curr_contagion=1:size(model.states)[2]
            # Get the budget and seeds for the current contagion.
            curr_budget = budgets[curr_contagion]
            curr_seeds = seed_set[curr_contagion]
            # Set the number of blocking for the current contagion to 0.
            curr_blocked = 0
            j=1
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
        blockers = Vector{Set{Int}}(undef, length(budgets))
        for i=1:length(blockers)
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

    function mcich(model, seed_sets::Tuple, updates:: Array, budgets::Vector{Int}, min_time=1, max_time = Nothing)
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
        num_contagions  = size(model.states)[2]
        # Initialize an object to store the blocking nodes in.
        blockings = Vector{Set}(undef, num_contagions)
        # Initialize the blocking point to 1 for each contagion.
        blocking_points = ones(num_contagions)
        # Set the maximum time to the set of nodes before the last set of infected nodeStates
        # if it is not already set.
        if max_time == Nothing
            max_time = length(updates) - 1
        end
            for curr_contagion=1:length(budgets)
                # Get the current budget
                curr_budget = budgets[curr_contagion]
                # Get the current seed nodes
                curr_seed_nodes = seed_sets[curr_contagion]

                candidate_blocker, blocking_point = BlockingHelpers.compute_blocking_choice_mcich(model, updates, curr_contagion, curr_budget, curr_seed_nodes, min_time, max_time)
                # Update the blocking_points and the blocking choices.
                blocking_points[curr_contagion] = blocking_point
                blockings[curr_contagion] = candidate_blocker

                if length(blockings) < curr_budget && curr_contagion < length(curr_budgets)
                    budgets[curr_contagion + 1] += curr_budget - length(blockings)
                end
        end
        return blockings, blocking_points
    end



    function mcich(model, seed_sets::Tuple{Set{Int}, Set{Int}}, updates:: Vector{Tuple}, budgets::Vector{Int}, optimizer, min_time=1, max_time=Nothing)
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
        num_contagions  = size(model.states)[2]
        # Initialize an object to store the blocking nodes in.
        blockings = Vector{Set}(undef, num_contagions)
        # Initialize the blocking point to 1 for each contagion.
        blocking_points = ones(num_contagions)
        # Set the maximum time to the set of nodes before the last set of infected nodeStates
        # if it is not already set.
        if max_time == Nothing
            max_time = length(updates) - 1
        end
            for curr_contagion=1:length(budgets)
                # Get the current budget
                curr_budget = budgets[curr_contagion]
                # Get the current seed nodes
                curr_seed_nodes = seed_sets[curr_contagion]

                candidate_blocker, blocking_point = BlockingHelpers.compute_blocking_choice_mcich(model, updates, curr_contagion, curr_budget, curr_seed_nodes, optimizer, min_time, max_time)
                # Update the blocking_points and the blocking choices.
                blocking_points[curr_contagion] = blocking_point
                blockings[curr_contagion] = candidate_blocker

                if length(blockings) < curr_budget && curr_contagion < length(curr_budgets)
                    budgets[curr_contagion + 1] += curr_budget - length(blockings)
                end
        end
        return blockings, blocking_points
    end



    # function mcich(model, seed_sets::Tuple{Set{Int}, Set{Int}}, updates:: Vector{Tuple}, budgets::Vector{Int}, optimizer)
    #     blockings = Vector()
    #     for i=1:length(budgets)
    #         budget = budgets[i]
    #         candidate_blocker = Vector{Int}()
    #         unblocked_min = Inf
    #         seed_nodes = seed_sets[i]
    #         for j=1:(length(updates) - 1)
    #             find_blocking = updates[j][i]
    #             if isempty(find_blocking)
    #                 break
    #             end
    #             available_to_block = setdiff(Set{Int}(keys(find_blocking)), seed_nodes)
    #             if length(available_to_block) <= budget
    #                 candidate_blocker = available_to_block
    #                 break
    #             end
    #             to_block = Dict{Int, UInt}()
    #             next_dict = updates[j+1][i]
    #             for node in available_to_block
    #                 for neighbor in all_neighbors(model.network, node)
    #                     if haskey(next_dict, neighbor)
    #                         requirement = get(next_dict, neighbor, 0) - get(model.thresholdStates, neighbor, model.θ_i[i]) + 1
    #                         get!(to_block, neighbor, requirement)
    #                     end
    #                 end
    #             end
    #             if isempty(to_block)
    #                 break
    #             end
    #             array_version = Vector{Int}(undef, length(available_to_block))
    #             cnt = 1
    #             for node in available_to_block
    #                 array_version[cnt] = node
    #                 cnt += 1
    #             end
    #             current_blocking, unblocked = coverage_optimal(model, array_version, to_block, budget, optimizer)
    #             if unblocked == 0
    #                 candidate_blocker = current_blocking
    #                 break
    #             elseif unblocked < unblocked_min
    #                 candidate_blocker = current_blocking
    #                 unblocked_min = unblocked
    #             end
    #         end
    #     append!(blockings, [collect(candidate_blocker)])
    #     if length(blockings) < budget && i < length(budgets)
    #         budgets[i+1] += budget - length(blockings)
    #     end
    #     end
    #     return blockings
    # end




    function ilp_construction(model, seed_nodes::Tuple{Set{Int}, Set{Int}}, updates:: Vector{Tuple}, budget::Int, optimizer, net_vertices)
        # Optimizer says what solver to use
        lp = Model(optimizer)
        # Scrub out seed nodes


        num_vertices = length(net_vertices)

        x_vars = zeros(Int, num_vertices, 2)
        y_vars = zeros(Int, num_vertices, 2)
        z_vars = zeros(Int, num_vertices, 2)
        @variable(lp, x_vars[1:num_vertices, 1:2])
        @variable(lp, y_vars[1:num_vertices, 1:2])
        @variable(lp, z_vars[1:num_vertices, 1:2])

        for j=1:2
            for i=1:num_vertices
                set_binary(x_vars[i,j])
                set_binary(y_vars[i,j])
                set_binary(z_vars[i,j])
            end
        end

        for i=1:num_vertices
            node = net_vertices[i]

            if node in seed_nodes[1]
                @constraint(lp, y_vars[i,1] == 1)
            end
            if node in seed_nodes[2]
                @constraint(lp, y_vars[i,2] == 1)
            end
            neighbors_node = neighbors(model.network, node)
            node_degree = degree(model.network, node)
            neighbor_indices = Vector{Int}()
            for neighbor in neighbors_node
                append!(neighbor_indices, [findfirst(x->x==neighbor,net_vertices)])
            end
            for j=1:2
                @constraint(lp, node_degree*x_vars[i,j] + sum(y_vars[k, j] for k in neighbor_indices) <= node_degree + get(model.thresholdStates, node, model.θ_i[j]) - 1)
                @constraint(lp, x_vars[i, j] + y_vars[i, j] + z_vars[i, j] == 1)
            end
        end

        @constraint(lp, sum(z_vars[i, 1] + z_vars[i, 2] for i=1:num_vertices) <= budget)

        @objective(lp, Min, sum(y_vars[i, 1] + y_vars[i, 2] for i=1:num_vertices) - (length(seed_nodes[1])+ length(seed_nodes[2])) )

        return lp
    end

    function ilp_optimal(model, seed_nodes::Tuple{Set{Int}, Set{Int}}, updates:: Vector{Tuple}, budget::Int, optimizer)


        net_vertices = collect(Int, vertices(model.network))

        lp = ilp_construction(model, seed_nodes, updates, budget, optimizer, net_vertices)

        optimize!(lp)

        z_vars = lp[:z_vars]
        blockers = Dict{Int, UInt}()
        for j=1:2
            for i=1:length(net_vertices)
                if value.(z_vars[i, j]) == 1
                    state = get(blockers, net_vertices[i], 0)
                    state += j
                    delete!(blockers, net_vertices[i])
                    get!(blockers, net_vertices[i], state)
                end
            end
        end

        return blockers

    end

end
