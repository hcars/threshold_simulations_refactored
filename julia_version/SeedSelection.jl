module SeedSelection
    using LightGraphs;


    function choose_by_centola(model, num_seeds::Int)::Set{Int}
        chosen_vertex = rand(vertices(model.network))
        seeds = Set{Int}([chosen_vertex])
        while length(seeds) < num_seeds
            for neighbor in neighbors(model.network, chosen_vertex)
                union!(seeds, [neighbor])
                if length(seeds) >= num_seeds
                    return seeds
                end
            end
            chosen_vertex = rand(neighbors(model.network, chosen_vertex))
        end
        return seeds
    end
    
    function choose_random_k_core(model, k::Int, num_seeds::Int)::Set{Int}
        k_core_nodes = find_k_core(model.network, k)
        choices = Set{Int}()
        while length(choices) < num_seeds
            choice = rand(k_core_nodes)
            union!(choices, [choice])
        end
        return choices
    end
    
    
    function find_k_core(network, k::Int)::Set{Int}
        k_cores = Set{Int}()
        for node in vertices(network)
            if degree(network, node) >= k
                union!(k_cores, [node])
            end
        end
        return k_cores
    end


end