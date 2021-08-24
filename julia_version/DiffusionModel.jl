module DiffusionModel


using LightGraphs





mutable struct MultiDiffusionModel
    network::SimpleGraph
    nodeStates::Dict{Int,UInt}
    thresholdStates::Dict{Int,UInt32}
    blockedDict::Dict{Int,UInt}
    θ_i::Vector{UInt32}
	ξ_i::Vector{UInt8}
    t::UInt32
end

function MultiDiffusionModelConstructor(graph)
    nodeStates = Dict{Int,UInt}()
    blockedDict = Dict{Int,UInt}()
    thresholdStates = Dict{Int,UInt32}()
    return MultiDiffusionModel(
        graph,
        nodeStates,
        thresholdStates,
        blockedDict,
        [UInt32(2), UInt32(2)],
		[UInt8(0), UInt8(0)],
        UInt(0),
    )
end

function MultiDiffusionModelConstructor(graph, θ_i::Vector{UInt32})
    nodeStates = Dict{Int,UInt}()
    blockedDict = Dict{Int,UInt}()
    thresholdStates = Dict{Int,UInt32}()
    return MultiDiffusionModel(
        graph,
        nodeStates,
        thresholdStates,
        blockedDict,
        θ_i,
		[UInt8(0), UInt8(0)],
        UInt(0),
    )
end

function MultiDiffusionModelConstructor(graph, θ_i::Vector{UInt32}, ξ_i::Vector{UInt8})
    nodeStates = Dict{Int,UInt}()
    blockedDict = Dict{Int,UInt}()
    thresholdStates = Dict{Int,UInt32}()
    return MultiDiffusionModel(
        graph,
        nodeStates,
        thresholdStates,
        blockedDict,
        θ_i,
		ξ_i,
        UInt(0),
    )
end

function set_initial_conditions!(model::MultiDiffusionModel, seeds::Set{Int})
    nodeStates = Dict{Int,UInt}()
    model.blockedDict = Dict{Int,UInt}()
    for seed in seeds
	infection = rand(1:3)
        get!(nodeStates, seed, infection)
    end
    model.t = UInt32(0)
    model.nodeStates = nodeStates
end

function set_initial_conditions!(
    model::MultiDiffusionModel,
    seeds::Tuple{Set{Int},Set{Int}},
)
    nodeStates = Dict{Int,UInt}()
    model.blockedDict = Dict{Int,UInt}()
    for i = 1:length(seeds)
        for seed in seeds[i]
            value = get!(nodeStates, seed, 0)
            value += i

            delete!(nodeStates, seed)
            get!(nodeStates, seed, value)
        end
    end
    model.t = UInt32(0)
    model.nodeStates = nodeStates
end

function set_blocking!(model::MultiDiffusionModel, blockers::Vector)
    blockingDict = Dict{Int,UInt}()
    for i = 1:length(blockers)
        curr_set = blockers[i]
            # If node is in the ith vector, then, the add to the blocking dictionary.
        for node in curr_set
            state = get(blockingDict, node, 0)
            state += i
            delete!(blockingDict, node)
            get!(blockingDict, node, state)
        end
    end
    model.blockedDict = blockingDict
end

function iterate!(model::MultiDiffusionModel)::Tuple
    """
    This completes a one time step update.
    """
    updated_1 = Dict{Int,UInt32}()
    updated_2 = Dict{Int,UInt32}()
    for u in vertices(model.network)
        u_state = get(model.nodeStates, u, 0)
        u_blocked_state = get(model.blockedDict, u, 0)
        if (u_state != 3)
            cnt_infected_1 = UInt32(0)
            cnt_infected_2 = UInt32(0)
            for v in all_neighbors(model.network, u)
                if (get(model.nodeStates, v, 0) == 1 ||
                    get(model.nodeStates, v, 0) == 3)
                    cnt_infected_1 += 1
                end
                if (get(model.nodeStates, v, 0) == 2 ||
                    get(model.nodeStates, v, 0) == 3)
                    cnt_infected_2 += 1
                end
            end
			interaction_term_1_2 = 0
			if (u_state == 2)
				interaction_term_1_2 = model.ξ_i[1]
			end
			interaction_term_2_1 = 0
			if (u_state == 1)
				interaction_term_2_1 = model.ξ_i[2]
			end

            thres_1 = get(
                model.thresholdStates,
                u,
                model.θ_i[1],
            ) - interaction_term_1_2
            transition_1 = (cnt_infected_1 >= thres_1) &&
                           ((u_state != 1) &&
                            (u_state != 3) && (u_blocked_state != 1) && (u_blocked_state != 3))
            thres_2 = get(
                model.thresholdStates,
                u,
                model.θ_i[2],
            ) - interaction_term_2_1
            transition_2 = (cnt_infected_2 >= thres_2) &&
                           ((u_state != 2) &&
                            (u_state != 3) && (u_blocked_state != 2) && (u_blocked_state != 3))

            if (transition_1 == true)
                get!(updated_1, u, cnt_infected_1)
            end
            if (transition_2 == true)
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
        state_summary[nodeState+1] += 1
    end
    state_summary[1] = nv(model.network) -
                       sum((
        state_summary[2],
        state_summary[3],
        state_summary[4],
    ))
    return state_summary
end



function full_run(model::MultiDiffusionModel)
    updates = Vector{Tuple}()
    updated = iterate!(model)
    max_infections = nv(model.network)
    append!(updates, [updated])
    iter_count = 0
    while !(isempty(updated[1]) &&
            isempty(updated[2]) && iter_count < max_infections)
        updated = iterate!(model)
        if !(isempty(updated[1]) && isempty(updated[2]))
            append!(updates, [updated])
        end
        iter_count += 1
    end
    return updates
end


end
