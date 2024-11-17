"""The graph tasks to be tried with LLMs."""

import random

import networkx as nx

from . import graph_text_encoders


class GraphTask:
    """The parent class for all the graph tasks."""

    def __init__(self):
        self.name = "default"
        self.maximum_nnodes_cot_graph = 10

    def prepare_examples_dict(
        self,
        graphs: list[nx.Graph],
        generator_algorithms: list[str],
        encoding_method: str,
    ) -> dict[int, dict[str, str | list[int]]]:
        raise NotImplementedError()

    def create_few_shot_example(self, graph: nx.Graph, encoding_method: str, cot: bool):
        raise NotImplementedError()


class ShortestPath(GraphTask):
    """The graph task to check if there is a path from a source to target."""

    def __init__(self):
        super().__init__()
        self.name = "shortest_path"

    def prepare_examples_dict(
        self,
        graphs: list[nx.Graph],
        generator_algorithms: list[str],
        encoding_method: str,
    ) -> dict[int, dict[str, str | list[int]]]:
        # examples_dict = {}
        # name_dict = graph_text_encoders.get_tlag_node_encoder(None, encoding_method)

        for ind, graph in enumerate(graphs):
            source, target = random.sample(list(graph.nodes()), k=2)
            question = graph_text_encoders.encode_graph(graph, encoding_method)
            # task_description = (
            #    "Q: What is the length of the shortest path from node %s to node"
            #    " %s?\nA: "
            #    % (
            #        name_dict[source],
            #        name_dict[target],
            #    )
            # )
            # question += task_description
            try:
                path = nx.shortest_path(graph, source, target)
                answer = str(len(path) - 1) + "."
            except nx.NetworkXNoPath:
                #answer = "There is no path from node %s to node %s." % (
                #    name_dict[source],
                #    name_dict[target],
                #)
                answer = "No path."
                

    
        #    examples_dict[ind] = {
        #        "question": question,
        #        "answer": answer,
        #        "nnodes": str(len(graph.nodes())),
        #        "nedges": str(len(graph.edges())),
        #        "task_description": task_description,
        #        "graph": graph,
        #        "algorithm": generator_algorithms[ind],
        #        "node_ids": [source, target],
        #    }
        #return examples_dict