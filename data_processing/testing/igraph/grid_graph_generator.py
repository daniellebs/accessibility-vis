from contextlib import redirect_stderr

# from data_processing.neo4j_client import Neo4jClient, args
import json
import logging
import igraph
import numpy as np
# import pandas as pd
from timeit import default_timer as timer


class GraphGenerator:
    def __init__(self, number_of_rows, number_of_columns,
                 edge_probability=0.5, weight_mean=1, weight_std=0.2,
                 dry_run=False):

        self._number_of_rows = number_of_rows
        self._number_of_cols = number_of_columns
        self._edge_prob = edge_probability
        self._weight_mean = weight_mean
        self._weight_std = weight_std
        self._nodes = []
        self._edges = []
        self._dry_run = dry_run
        self._graph = igraph.Graph()
        self._nodes_to_index = dict()

    def bernoulli_trial(self):
        return np.random.random() > self._edge_prob

    def weight(self):
        # return np.random.normal(self._weight_mean, self._weight_std)
        return np.random.uniform(0.0, 3.0)

    def generate_planar_grid_graph(self):
        # Create cartesian product
        rows = range(self._number_of_rows)
        cols = range(self._number_of_cols)
        self._nodes = [(r, c) for r in rows for c in cols]

        self._nodes_to_index = {k: v for v, k in enumerate(self._nodes)}

        for i, j in self._nodes:
            curr_index = self._nodes_to_index[(i, j)]
            # Can also be done by iterating from (i-1,j-1) to (i+1,j+1).
            if i > 0 and j > 0 and self.bernoulli_trial():
                self._edges.append((curr_index,
                                    self._nodes_to_index[(i - 1, j - 1)],
                                    self.weight()))
            if i > 0 and self.bernoulli_trial():
                self._edges.append((
                    curr_index, self._nodes_to_index[(i - 1, j)],
                    self.weight()))
            if i > 0 and j < self._number_of_cols - 1 \
                    and self.bernoulli_trial():
                self._edges.append((curr_index,
                                    self._nodes_to_index[(i - 1, j + 1)],
                                    self.weight()))
            if j < self._number_of_cols - 1 and self.bernoulli_trial():
                self._edges.append((
                    curr_index, self._nodes_to_index[(i, j + 1)],
                    self.weight()))
            if i < self._number_of_rows - 1 and j < self._number_of_cols - 1 \
                    and self.bernoulli_trial():
                self._edges.append((curr_index,
                                    self._nodes_to_index[(i + 1, j + 1)],
                                    self.weight()))
            if i < self._number_of_rows - 1 and self.bernoulli_trial():
                self._edges.append((
                    curr_index, self._nodes_to_index[(i + 1, j)],
                    self.weight()))
            if i < self._number_of_rows - 1 and j > 0 \
                    and self.bernoulli_trial():
                self._edges.append((curr_index,
                                    self._nodes_to_index[(i + 1, j - 1)],
                                    self.weight()))
            if j > 0:
                self._edges.append((
                    curr_index, self._nodes_to_index[(i, j - 1)],
                    self.weight()))

        if not self._dry_run:
            start = timer()
            self._graph = igraph.Graph.TupleList(self._edges, weights=True)
            end = timer()
            tot_time = end - start
            print(igraph.summary(self._graph))
            print(f'Loading graph took {tot_time} seconds.')

    def get_number_of_nodes(self):
        return len(self._nodes)

    def get_number_of_edges(self):
        return len(self._edges)

    def get_shortest_paths(self, source):
        start = timer()

        sssp = self._graph.shortest_paths_dijkstra(source=source,
                                                   weights='weight',
                                                   mode=igraph.OUT)

        end = timer()
        tot_time = end - start
        num_nodes = self.get_number_of_nodes()
        num_edges = self.get_number_of_edges()
        num_sources = len(source)
        time_per_source_ms = (tot_time / num_sources) * 1000
        print(
            f'Computing SP from {num_sources} sources took {tot_time} seconds '
            f'for a graph with {num_nodes} nodes and {num_edges} edges.'
            f'This means {time_per_source_ms} ms per source.')
        return sssp

    # TODO(danielle): fix saving and loading graph
    # def save_graph(self):
    #     f = 'output/grid_graph.graphml'
    #     print(f'Saving graph to file {f}')
    #     self._graph.write_graphml(f)
    #
    # def load_graph(self):
    #     f = 'output/grid_graph.graphml'
    #     print(f'Loading graph from file {f}')
    #     self._graph.Read_GraphML(f)
    #     print(igraph.summary(self._graph))


if __name__ == '__main__':
    graph_generator = GraphGenerator(500, 500, 0.5, 1, 0.4)
    graph_generator.generate_planar_grid_graph()
    # graph_generator.save_graph()
    # TODO(danielle): write method to iterate nodes and get SP for all
    # graph_generator.load_graph()
    sssp = graph_generator.get_shortest_paths(list(range(100)))
    # print(sssp)
