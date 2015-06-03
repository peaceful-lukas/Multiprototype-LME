function val = is_connected_graph(A)

    numVertices = size(A, 1);
    visited = zeros(1, numVertices);
    unvisited_vertices = 1:numVertices;

    visited = depth_first_traverse(A, visited, 2);
    
    if sum(visited) < numVertices
        val = 0;
    else
        val = 1;
    end
end


function visited = depth_first_traverse(A, visited, v)
    
    visited(v) = 1;
    next = find(A(v, :));

    for n = next
        if visited(n) == 1
            continue;
        else
            visited = depth_first_traverse(A, visited, n);
        end
    end    
end