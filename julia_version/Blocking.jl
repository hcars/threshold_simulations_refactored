
module Blocking

    using LightGraphs;
    include("./DiffusionModel.jl")
    function mcich(model, seed_nodes::Set{Int}, updates:: Vector{Tuple}, budgets::Vector{Int})
        blockings = Vector()
        for i=1:length(budgets)
            budget = budgets[i]
            candidate_blocker = Vector{Int}()
            unblocked_min = Inf 
            for j=1:length(updates)
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
                current_blocking, unblocked = greedy_smc(model, Set{Int}(available_to_block), to_block, budget)
                if unblocked == 0
                    candidate_blocker = current_blocking 
                    break
                elseif unblocked < unblocked_min
                    candidate_blocker = current_blocking
                end
            end
        append!(blockings, [candidate_blocker])
        end
        return blockings
    end

    function greedy_smc(model, available_to_block::Set{Int}, to_block::Dict{Int, UInt}, budget::Int)
        upperLimit = minimum([budget, length(available_to_block)])
        best_blocking = Vector{Int}(undef, upperLimit)
        for i=1:upperLimit
            largest_intersection = 0
            local best_node::Int
            for possible_node in available_to_block
                neighbors_in_next = Set{Int}()
                for neighbor in all_neighbors(model.network, possible_node)
                    if haskey(to_block, neighbor)
                        union!(neighbors_in_next, [neighbor])
                    end
                end
                unblocked_neighbors = intersect(neighbors_in_next, keys(to_block))
                if length(unblocked_neighbors) > largest_intersection
                    largest_intersection = length(unblocked_neighbors)
                    best_node = possible_node
                end
            end
            setdiff!(available_to_block, [best_node])
            best_blocking[i] = best_node
            for node in neighbors(model.network, best_node)
                if haskey(to_block, node)
                    requirement = get(to_block, node, 0)
                    requirement -= 1
                    delete!(to_block, node)
                    if requirement > 0 
                        get!(to_block, node, requirement)
                    end
                end
            end
        end
        return best_blocking, length(keys(to_block))
    end


end
