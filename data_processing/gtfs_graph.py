import igraph
from multiprocessing import Pool
import numpy as np
import pickle
import pandas as pd
from time import time
import os
from timeit import default_timer as timer
import datetime as dt
import tqdm
import pyarrow


class GtfsGraph:
    def __init__(self, direct_edges, transfer_edges={}, nodes_df=None, reversed_graph=False):
        self._edges = [(str(u), str(v), w) for u,v, w in direct_edges.union(transfer_edges)]
        nodes = [(s, d) for s, d, _ in self._edges]
        self._nodes = list(set([item for sublist in nodes for item in sublist]))
        self._graph = igraph.Graph()
        self._nodes_df = nodes_df
        print("========== Nodes DataFrame ==========")
        print(self._nodes_df.head(3))
        print("=====================================")

    def construct_graph(self):
        s = timer()
        self._graph = igraph.Graph.TupleList(self._edges, weights=True,
                                             directed=True)

        e = timer()
        print(igraph.summary(self._graph))
        print(f'Loading graph took {e - s} seconds.')

    def get_number_of_nodes(self):
        return len(self._nodes)

    def get_number_of_edges(self):
        return len(self._edges)

    def get_nodes(self):
        return self._nodes

    def get_shortest_paths(self, sources, save_to_files=True, debug=False, target=None):
        if not debug:
            sp = self._graph.shortest_paths_dijkstra(source=sources,
                                                     target=self._nodes,
                                                     weights='weight',
                                                     mode=igraph.OUT)

            max_len_sec = 3600  # 1 hour
            reachable = {'source': [], 'target': [], 'time_sec': []}
            for i, source in enumerate(sources):
                # Verify all results and correct length
                assert len(self._nodes) == len(sp[i])
                dists = sp[i]

                # Go through all targets and save only reachable nodes
                for target_i, d in enumerate(dists):
                    if d <= max_len_sec:
                        reachable['source'].append(int(source))
                        reachable['target'].append(int(self._nodes[target_i]))
                        reachable['time_sec'].append(d)

            reachable_df = pd.DataFrame.from_dict(reachable)
            del reachable  # Delete reachable dict from memory

            # print("======== Reachable DataFrame ========")
            # print(reachable_df.head(3))
            # print("=====================================")

            reachable_df = reachable_df.merge(
                self._nodes_df, left_on='target', right_on='node_id').drop(['target', 'node_id', 'departure'], axis=1)
            reachable_df = reachable_df.merge(
                self._nodes_df, left_on='source', right_on='node_id', suffixes=('_target', '_source')).drop(['source', 'node_id', 'arrival_source'], axis=1)
            reachable_df = reachable_df.sort_values('time_sec').groupby(['stop_id_source','stop_id_target'], as_index=False).first()

            if save_to_files:
                filename = '../output_data/sp/sp_' + str(
                    sources[0]) + '-' + str(
                    sources[-1]) + '.pkl'
                reachable_df.to_pickle(filename)


        if debug:
            print(f'Trying to get a shortest path from {sources} to {target}')
            assert target is not None, 'In debug mode the target must be set'
            sp = self._graph.get_shortest_path_dijkstra(sources, to=target,
                                                weights='weight',
                                                mode=igraph.OUT)
            print('======================')
            print(f'Path to node {sp[-1]} is {sp}')


def batches(l, n):
    """Yield successive n-sized batches from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


START_NODES_PATH = '../output_data/morning_start_nodes.pkl'
ALL_NODES_PATH = '../output_data/morning_nodes.pkl'
DIRECT_EDGES_PATH = '../output_data/morning_direct_edges.pkl'
# DIRECT_EDGES_PATH = '../output_data/single_trip_direct_edges.pkl'
TRANSFER_EDGES_PATH = '../output_data/morning_transfer_edges.pkl'
DEBUG = False

if __name__ == '__main__':
    direct_edges = pd.read_pickle(DIRECT_EDGES_PATH)
    transfer_edges = pd.read_pickle(TRANSFER_EDGES_PATH)
    all_nodes_df = pd.read_pickle(ALL_NODES_PATH)[
        ['node_id', 'stop_id', 'stop_lon', 'stop_lat', 'departure', 'arrival']]
    gtfs_graph = GtfsGraph(direct_edges, transfer_edges, all_nodes_df)
    gtfs_graph.construct_graph()
    print('Finished constructing the graph')

    with open(START_NODES_PATH, 'rb') as f:
        nodes = pickle.load(f)
        graph_nodes = set(gtfs_graph.get_nodes())
        nodes = [n for n in nodes if n in graph_nodes]

    num_of_sources = len(nodes)
    batch_size = 100
    pool_input = batches(nodes, batch_size)

    start = timer()
    p = Pool()
    p.map(gtfs_graph.get_shortest_paths, pool_input)
    p.close()
    p.join()
    end = timer()
    pool_tot_time = end - start
    time_for_source = (end - start) / num_of_sources
    print("Pool took: ", pool_tot_time, ", so ", time_for_source, " per source")
