import igraph
from multiprocessing import Pool
import numpy as np
import pickle
from time import time
from timeit import default_timer as timer


class GraphGenerator:
    def __init__(self, number_of_rows, number_of_columns,
                 edge_probability=0.5, weight_low=0, weight_high=3,
                 dry_run=False):
        self._number_of_rows = number_of_rows
        self._number_of_cols = number_of_columns
        self._edge_prob = edge_probability
        self._weight_low = weight_low
        self._weight_high = weight_high
        self._nodes = []
        self._edges = []
        self._dry_run = dry_run
        self._graph = igraph.Graph()
        self._nodes_to_index = dict()

    def bernoulli_trial(self):
        return np.random.random() < self._edge_prob

    def weight(self):
        return np.random.uniform(self._weight_low, self._weight_high)

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

    def get_shortest_paths(self, sources, save_to_files=False):
        sp = self._graph.shortest_paths_dijkstra(source=sources,
                                                 weights='weight',
                                                 mode=igraph.OUT)
        # if save_to_files:
        #     filename = 'sp_' + str(sources[0]) + '-' + str(sources[-1]) + '.pkl'
        #     with open(filename, 'wb') as f:
        #         # store the data as binary data stream
        #         pickle.dump(sp, f, pickle.HIGHEST_PROTOCOL)

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


def batches(l, n):
    """Yield successive n-sized batches from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':
    graph_generator = GraphGenerator(number_of_rows=500, number_of_columns=500,
                                     edge_probability=0.7, weight_low=0,
                                     weight_high=4)
    graph_generator.generate_planar_grid_graph()

    parallelize = True

    # graph_generator.save_graph()
    # graph_generator.load_graph()
    # TODO(danielle): write method to iterate nodes and get SP for all

    # Single step computation
    batch_size = 100
    print(f'batch size: {batch_size}')
    start = time()
    graph_generator.get_shortest_paths(list(range(200, 300)))
    end = time()
    # pickle_in = open("sp_test.data", "rb")
    # sp = pickle.load(pickle_in)
    time_for_batch = end - start
    print("Single step took: ", time_for_batch, ", so ",
          time_for_batch / batch_size, " per source")

    # Let's try using a pool
    num_of_sources = 2000
    print(f'number of sources: {num_of_sources}')
    start = timer()
    p = Pool()

    pool_input = batches(list(range(num_of_sources)), batch_size)
    p.map(graph_generator.get_shortest_paths, pool_input)

    p.close()
    p.join()
    end = timer()
    pool_tot_time = end - start
    time_for_source = (end - start) / num_of_sources
    print("Pool took: ", pool_tot_time, ", so ", time_for_source, " per source")
