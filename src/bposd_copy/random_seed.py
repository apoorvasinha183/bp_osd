import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np 
def generate_biregular_graph(M, N, d1, d2, max_attempts=100):
    # Check preconditions
    if M * d1 != N * d2 or M <= 0 or N <= 0 or d1 <= 0 or d2 <= 0 or max_attempts <= 0:
        raise ValueError("Invalid parameters. M * d1 must equal N * d2, and all parameters must be positive.")

    # Attempt to generate a valid biregular bipartite graph
    for attempt in range(1, max_attempts + 1):
        try:
            # Generate bipartite graph using configuration model
            G = nx.bipartite.configuration_model([d1] * M, [d2] * N, create_using=nx.Graph)

            # Validate the generated graph
            if is_valid_biregular_bipartite_graph(G, M, N, d1, d2):
                print(f"Valid biregular bipartite graph generated after {attempt} attempts.")
                return G

        except ValueError:
            continue  # Ignore any exceptions and proceed to the next attempt

    # Raise an error if no valid graph is generated after all attempts
    raise ValueError(f"Failed to generate a valid biregular bipartite graph after {max_attempts} attempts.")

def is_valid_biregular_bipartite_graph(G, M, N, d1, d2):
    # Check if the graph has the correct number of nodes and edges
    if len(G) != M + N or G.size() != M * d1:
        return False

    # Validate degrees of nodes in partition 1
    for node in range(M):
        if G.degree[node] != d1:
            return False

    # Validate degrees of nodes in partition 2
    for node in range(M, M + N):
        if G.degree[node] != d2:
            return False

    return True

def bipartite_to_adjacency_matrix(graph, M):
    """
    Convert a bipartite graph into an adjacency matrix suitable for LDPC representation.

    Parameters:
    - graph: NetworkX bipartite graph object.
    - M: Number of vertices in the first partition (data nodes).

    Returns:
    - A NumPy array representing the adjacency matrix.
    """
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = np.zeros((M, len(graph) - M), dtype=int)

    # Iterate over data nodes and their neighbors (check nodes)
    for data_node in range(M):
        neighbors = list(graph.neighbors(data_node))
        for check_node in neighbors:
            adjacency_matrix[data_node, check_node - M] = 1

    return adjacency_matrix.T # This is how the library expects the code to be


