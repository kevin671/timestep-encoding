import networkx as nx


def encode_graph(
    graph: nx.Graph, graph_encoder=None, node_encoder=None, edge_encoder=None
) -> str:
    r"""Encodes a graph as text.

    This relies on choosing:
       a node_encoder and an edge_encoder:
       or
       a graph_encoder (a predefined pair of node and edge encoding strategies).

    Note that graph_encoders may assume that the graph has some properties
    (e.g. integer keys).

    Example usage:
    .. code-block:: python
    ```
    # Use a predefined graph encoder from the paper.
    >>> G = nx.karate_club_graph()
    >>> encode_graph(G, graph_encoder="adjacency")
    'In an undirected graph, (i,j) means that node i and node j are
    connected
    with an undirected edge. G describes a graph among nodes 0, 1, 2, 3, 4, 5,
    6,
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, and 33.\nThe edges in G are: (0, 1) (0, 2) (0, 3)
    ...'

    # Use the node's name in the graph as the node identifier.
    >>> G = nx.les_miserables_graph()
    >>> encode_graph(G, node_encoder="nx_node_name", edge_encoder="friendship")
    'G describes a friendship graph among nodes Anzelma, Babet, Bahorel,
    Bamatabois, BaronessT, Blacheville, Bossuet, Boulatruelle, Brevet, ...
    We have the following edges in G:
    Napoleon and Myriel are friends. Myriel and MlleBaptistine are friends...'

    # Use the `id` feature from the edges to describe the edge type.
    >>> G = nx.karate_club_graph()
    >>> encode_graph(G, node_encoder="nx_node_name", edge_encoder="nx_edge_id")
    'In an undirected graph, (s,p,o) means that node s and node o are connected
    with an undirected edge of type p. G describes a graph among nodes 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, and 33.
    The edges in G are: (0, linked, 1) (0, linked, 2) (0, linked, 3) ...'
    ```

    Args:
      graph: the graph to be encoded.
      graph_encoder: the name of the graph encoder to use.
      node_encoder: the name of the node encoder to use.
      edge_encoder: the name of the edge encoder to use.

    Returns:
      The encoded graph as a string.
    """

    # Check that only one of graph_encoder or (node_encoder, edge_encoder) is set.
    if graph_encoder and (node_encoder or edge_encoder):
        raise ValueError(
            "Only one of graph_encoder or (node_encoder, edge_encoder) can be set."
        )

    if graph_encoder:
        if isinstance(graph_encoder, str):
            node_encoder_dict = get_tlag_node_encoder(graph, graph_encoder)
            return EDGE_ENCODER_FN[graph_encoder](graph, node_encoder_dict)
        else:
            return graph_encoder(graph)

    else:
        node_encoder_dict = nodes_to_text(graph, node_encoder)
        return EDGE_ENCODER_FN[edge_encoder](graph, node_encoder_dict)
