import igraph
from multiprocessing import Pool
import numpy as np
import pickle
import pandas as pd
from time import time
import os
from timeit import default_timer as timer


class GtfsGraph:
    def __init__(self, direct_edges, transfer_edges={}, reversed_graph=False):
        # TODO: Re-enumerate nodes (also in edges) so results will make sense
        self._nodes = []
        if reversed_graph:
            direct_edges = {(d, s, w) for s, d, w in direct_edges}
            transfer_edges = {(d, s, w) for s, d, w in transfer_edges}
        self._edges = direct_edges.union(transfer_edges)

        # Re-enumerate nodes to simplify working with shortest-path algorithm
        # results.
        nodes = [(s, d) for s, d, _ in self._edges]
        nodes = list(set([item for sublist in nodes for item in sublist]))
        print(f'length of nodes: {len(nodes)}')
        nodes_enumeration = list(enumerate(nodes))
        print(f'length of nodes enumeration: {len(nodes_enumeration)}')
        self._index_to_node = dict(nodes_enumeration)
        self._node_to_index = {n: i for i, n in nodes_enumeration}
        print(f'length of nodes to indes: {len(self._node_to_index)}')

        # Apply nodes enumeration to edges.
        self._edges = {(self._node_to_index[s], self._node_to_index[d], w) for
                       s, d, w in self._edges}

        self._graph = igraph.Graph()

    def construct_graph(self):
        start = timer()
        self._graph = igraph.Graph.TupleList(self._edges, weights=True)
        # TODO: Get all graph nodes and save enumeration for the SP results
        end = timer()
        tot_time = end - start
        print(igraph.summary(self._graph))
        print(f'Loading graph took {tot_time} seconds.')

    def get_number_of_nodes(self):
        return len(self._nodes)

    def get_number_of_edges(self):
        return len(self._edges)

    def get_shortest_paths(self, sources, save_to_files=True):
        sp = self._graph.shortest_paths_dijkstra(source=sources,
                                                 weights='weight',
                                                 mode=igraph.OUT)
        print([x for x in sp[0] if x < np.inf])


        # TODO: Since this is miltiprocessed we don't share the memory. We must find another way to store results.
        sp_results = {}
        num_destinations = None
        for i, source in enumerate(sources):
            if i == 0:
                num_destinations = len(sp[i])
            # Verify all results have the same length
            assert num_destinations == len(sp[i])
            sp_results[source] = sp[i]

        if save_to_files:
            filename = '../output_data/sp/sp_' + str(sources[0]) + '-' + str(
                sources[-1]) + '.pkl'
            with open(filename, 'wb') as f:
                # store the data as binary data stream
                pickle.dump(sp_results, f, pickle.HIGHEST_PROTOCOL)

    def get_reachable_nodes(self, time_limit=900):
        # Time limit of 900 seconds means 15 minutes
        # Read all pkl files in ../output_data/sp and gather results

        results = dict()
        for f in os.listdir('../output_data/sp/'):
            results.update(pickle.load(open(f, 'rb')))

        print(f'length of results at the beginning is {len(results)}')
        reachable_index = results.copy()
        reachable = dict()
        for s, sp in results.items():
            # print(f'source is {s}')
            for i in range(len(sp)):
                assert len(reachable_index[s]) > i, f'source {s} has {len(reachable_index[s])} but now iterating with index {i}'
                if sp[i] > time_limit:
                    #print(f'the length of reachable in current source is {len(reachable[s])} and index is {i}')
                    reachable_index[s][i] = np.inf

            # Get nodes of indexes of reachable for each source
            reachable[self._index_to_node[s]] = [self._index_to_node[i] for i in
                                                 range(len(reachable_index[s])) if
                                                 reachable_index[s][i] < np.inf]
        print(f'length of reachable at the end is {len(reachable)}')
        return reachable

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


NODES_PATH = '../input_data/pickles/morning_trips_nodes.pkl'
DIRECT_EDGES_PATH = '../input_data/pickles/direct_morning_edges_all_israel.pkl'

if __name__ == '__main__':
    nodes_df = pd.read_pickle(NODES_PATH)
    direct_edges = pd.read_pickle(DIRECT_EDGES_PATH)
    print(list(direct_edges)[:5])  # TODO: remove
    gtfs_graph = GtfsGraph(direct_edges)
    gtfs_graph.construct_graph()

    num_of_sources = 100  # TODO: Fix to actual size of the data
    batch_size = 100

    print(f'number of sources: {num_of_sources}')
    start = timer()
    p = Pool()

    pool_input = batches(list(range(num_of_sources)), batch_size)
    p.map(gtfs_graph.get_shortest_paths, pool_input)

    p.close()
    p.join()
    end = timer()
    pool_tot_time = end - start
    time_for_source = (end - start) / num_of_sources
    print("Pool took: ", pool_tot_time, ", so ", time_for_source, " per source")
    reachable = gtfs_graph.get_reachable_nodes()
    print(len(reachable))
    # TODO: map node indexes we found to actual nodes
    for s in reachable:
        print(f'source {s} can reach {reachable[s]}')
        print(nodes_df.iloc[s])
        print(nodes_df.iloc[reachable[s][0]])
