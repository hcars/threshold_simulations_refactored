module Blocking
    using LightGraphs;
    using JuMP;
    include("./DiffusionModel.jl")


    function high_degree_blocking(model, budgets::Vector{Int}, seed_set::Tuple)::Vector{Set{Int}}
        my_degree(x) = degree(model.network, x)
        high_degree_nodes = sort(vertices(model.network), by=my_degree, rev=true)
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

    function mcich(model, seed_sets::Tuple{Set{Int}, Set{Int}}, updates:: Vector{Tuple}, budgets::Vector{Int})
        blockings = Vector()
            blocking_point = [1,1]

            for i=1:length(budgets)
                budget = budgets[i]
                candidate_blocker = Vector{Int}()
                unblocked_min = Inf 
                seed_nodes = seed_sets[i]
                for j=1:length(updates) - 1
                    find_blocking = updates[j][i]
                    if isempty(find_blocking)
                        break
                    end
		    # Here we are differencing the seed nodes out from the seed set.
                    # This could be a source of randomness since the set object is unordered.
                    available_to_block = setdiff(Set{Int}(keys(find_blocking)), seed_nodes)
                    if length(available_to_block) <= budget
                        candidate_blocker = available_to_block
                        blocking_point[i] = j

                        break
                    end
                    to_block = Dict{Int, UInt}()
                    next_dict = updates[j+1][i]
                    blocking_map = Dict{Int, Vector}()
                    for node in available_to_block
                        neighbors_to_block = Vector{Int}()
                        for neighbor in all_neighbors(model.network, node)
                            if haskey(next_dict, neighbor)
                                requirement = get(next_dict, neighbor, 0) - get(model.thresholdStates, neighbor, model.θ_i[i]) + 1
                                get!(to_block, neighbor, requirement)
                                append!(neighbors_to_block, [neighbor])
                            end
                        end
                        get!(blocking_map, node, neighbors_to_block)
                    end
                    if isempty(to_block)
                        break
                    end
                    current_blocking, unblocked = coverage(blocking_map, to_block, budget)
                    if unblocked == 0
                        candidate_blocker = current_blocking 
                        blocking_point[i] = j 
                        break
                    elseif unblocked < unblocked_min
                        candidate_blocker = current_blocking
                        unblocked_min = unblocked   
                        blocking_point[i]  = j
                    end
                end  
            append!(blockings, [collect(candidate_blocker)])
            if length(blockings) < budget && i < length(budgets)
                budgets[i+1] += budget - length(blockings)
            end
        end
        return blockings, blocking_point
    end

    

    function coverage(available_to_block::Dict{Int, Vector}, to_block::Dict{Int, UInt}, budget::Int)
        upperLimit = minimum([budget, length(available_to_block)])
        best_blocking = Vector{Int}(undef, upperLimit)
        for i=1:upperLimit
            possible_blocking_nodes = collect(keys(available_to_block)) 

            intersections = map(x->length(intersect(get(possible_blocking_nodes, x, []), keys(to_block))), keys(possible_blocking_nodes))

	    best_node = possible_blocking_nodes[argmax(intersections)] 
	    best_blocking[i] = best_node 

            for node in get(available_to_block, best_node, [])
                if node in keys(to_block)

                    requirement = get(to_block, node, 0)
                    requirement -= 1
                    delete!(to_block, node)

                    if requirement > 0 
                        get!(to_block, node, requirement)
                    end
                end
            end
            delete!(available_to_block, best_node)

            if isempty(to_block)
                break
            end
        end

        return best_blocking, length(keys(to_block))
    end

    function mcich_optimal(model, seed_sets::Tuple{Set{Int}, Set{Int}}, updates:: Vector{Tuple}, budgets::Vector{Int}, optimizer)
        blockings = Vector()
        for i=1:length(budgets)
            budget = budgets[i]
            candidate_blocker = Vector{Int}()
            unblocked_min = Inf 
            seed_nodes = seed_sets[i]
            for j=1:(length(updates) - 1)
                find_blocking = updates[j][i]
                if isempty(find_blocking)
                    break
                end
                available_to_block = setdiff(Set{Int}(keys(find_blocking)), seed_nodes)
                if length(available_to_block) <= budget
                    candidate_blocker = available_to_block
                    break
                end
                to_block = Dict{Int, UInt}()
                next_dict = updates[j+1][i]
                for node in available_to_block
                    for neighbor in all_neighbors(model.network, node)
                        if haskey(next_dict, neighbor)
                            requirement = get(next_dict, neighbor, 0) - get(model.thresholdStates, neighbor, model.θ_i[i]) + 1
                            get!(to_block, neighbor, requirement)
                        end
                    end
                end
                if isempty(to_block)
                    break
                end
                array_version = Vector{Int}(undef, length(available_to_block))
                cnt = 1
                for node in available_to_block
                    array_version[cnt] = node
                    cnt += 1
                end
                current_blocking, unblocked = coverage_optimal(model, array_version, to_block, budget, optimizer)
                if unblocked == 0
                    candidate_blocker = current_blocking 
                    break
                elseif unblocked < unblocked_min
                    candidate_blocker = current_blocking
                    unblocked_min = unblocked
                end
            end
        append!(blockings, [collect(candidate_blocker)])
        if length(blockings) < budget && i < length(budgets)
            budgets[i+1] += budget - length(blockings)
        end
        end
        return blockings
    end


    function coverage_optimal(model, available_to_block::Array{Int}, to_block::Dict{Int, UInt}, budget::Int, optimizer)
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
        blockers = Vector{Int}()
        for (index, y) in enumerate(y_j)
            if value.(y) == 1
                append!(blockers, [available_to_block[index]])
            end
        end
        return blockers, number_to_block - objective_value(lp)
        else
           println(termination_status(lp))
           error("The model did not solve or run out of time.")
        end
    end

    function ilp_optimal(model, seed_nodes::Set{Int}, updates:: Vector{Tuple}, budget::Int, optimizer)
        lp = Model(optimizer)
        # Scrub out seed nodes
        for update_t in updates
            for update_set in update_t
                for node in keys(update_set)
                    if node in seed_nodes
                        pop!(update_set, node, 0)
                    end
                end
            end
        end
        net_vertices = collect(Int, vertices(model.network))
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
            state = get!(model.nodeStates, node, 0)
            if node in seed_nodes
                if state == 1
                    @constraint(lp, y_vars[i,1] == 1)
                elseif  state == 2
                    @constraint(lp, y_vars[i,2] == 1)
                elseif state == 3
                    @constraint(lp, y_vars[i,1] == 1)
                    @constraint(lp, y_vars[i,2] == 1)
                end
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
        @objective(lp, Min, sum(y_vars[i, 1] + y_vars[i, 2] for i=1:num_vertices))
        optimize!(lp)

        blockers = Dict{Int, UInt}()
        for j=1:2
            for i=1:num_vertices
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
