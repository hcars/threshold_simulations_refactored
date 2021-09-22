module Optimal_ILP_Helpers
    using LightGraphs;
    using JuMP;

    function optimal_ilp_construction(model, seed_nodes::Tuple, updates::Vector, budget::Int, optimizer, net_vertices)
        """
        This constructs that ILP for the blocking problem.
        """

        # Optimizer says what solver to use
        lp = Model(optimizer)
        # Get the number of contagions and number of verices.
        num_contagions = size(model.states)[2]
        num_vertices = length(net_vertices)
        # Initialize the matrices with the variables.
        x_vars = zeros(Int, num_vertices, num_contagions)
        y_vars = zeros(Int, num_vertices, num_contagions)
        z_vars = zeros(Int, num_vertices, num_contagions)
        @variable(lp, x_vars[1:num_vertices, 1:num_contagions])
        @variable(lp, y_vars[1:num_vertices, 1:num_contagions])
        @variable(lp, z_vars[1:num_vertices, 1:num_contagions])

        for curr_contagion=1:num_contagions
            # Set the seed nodes to infected.
            @constraint(lp, y_vars[collect(seed_nodes[curr_contagion]), curr_contagion] .== 1)
            for curr_vertex=1:num_vertices
                # Set variables to binary.
                set_binary(x_vars[curr_vertex,curr_contagion])
                set_binary(y_vars[curr_vertex,curr_contagion])
                set_binary(z_vars[curr_vertex,curr_contagion])
            end
        end


        for curr_vertex=1:num_vertices
            # Get the current node ID.
            node = net_vertices[curr_vertex]
            # Get the neighbor of the current node.
            neighbors_node = neighbors(model.network, node)
            # Get the degree of the current node.
            node_degree = degree(model.network, node)
            # Set the constraints that will set the nodes to 0 appropriately.
            for curr_contagions=1:num_contagions
                @constraint(lp, node_degree * x_vars[curr_vertex, curr_contagions] + sum(y_vars[neighbors_node, curr_contagions]) <= node_degree + model.thresholds[node, curr_contagions] - 1)
            end
            @constraint(lp, x_vars[curr_vertex, :] .+ y_vars[curr_vertex, :] .+ z_vars[curr_vertex, :] .== 1)

        end
        # Add constratin that manages the budget.
        @constraint(lp, sum(z_vars) <= budget)
        # Set the objective.
        @objective(lp, Min, sum(y_vars) - sum(map(x->length(x), seed_nodes)))

        return lp
    end

    function optimal_ilp_solutions(model, lp, net_vertices)
        """
        Get the ILP solutions into the same format as the other blocking methods.
        """
        # Check the termination status of the model.
        if (termination_status(lp) == MOI.TIME_LIMIT) || (termination_status(lp) == MOI.OPTIMAL)


            z_vars = lp[:z_vars]
            num_contagions = size(model.states)[2]
            num_vertices = nv(model.network)
            blockers = Vector{Set}(undef, num_contagions)
            for curr_contagion=1:num_contagions
                curr_blockers = Set{Int}()
                for curr_vertex=1:num_vertices
                    # Find and join the blocking nodes to the appropriate blocking set.
                    if value.(z_vars[curr_vertex, curr_contagion]) == 1
                        union!(curr_blockers, [net_vertices[curr_vertex]])
                    end
                end
                # Add the set to the vector of blockers.
                blockers[curr_contagion] = curr_blockers
            end
            return blockers
        else
           println(termination_status(lp))
           error("The model has a bad termination status.")
        end
    end


end
