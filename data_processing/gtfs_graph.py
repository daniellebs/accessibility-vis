import igraph
from multiprocessing import Pool
import pickle
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from time import sleep
import argparse

# Command line flags
parser = argparse.ArgumentParser()
# TODO: Add flags here
args = parser.parse_args()


class GtfsGraph:
    def __init__(self, direct_edges, transfer_edges={}, nodes_df=None,
                 reversed_graph=False):
        self._edges = self.construct_edges(direct_edges, transfer_edges)
        self._nodes = self.construct_nodes()
        self._graph = igraph.Graph()
        self._nodes_df = nodes_df
        self._reversed = reversed_graph
        self._direct_edges = set()

    def create_direct_edges(self, raw_nodes_df):
        # Sanity check: verify that the stops in each trip are consecutive
        def verify_consecutive(l):
            if sorted(list(l)) != list(range(min(l), max(l) + 1)):
                print('Found non-consecutive stop sequence')
                raise Exception('NON CONSECUTIVE STOP SEQUENCE')

        tmp_df = raw_nodes_df
        tmp_df[['trip_id', 'stop_sequence']].groupby('trip_id').apply(
            lambda x: verify_consecutive(x.stop_sequence))
        del tmp_df

        # Create direct edges
        # TODO: Consider making this part parallel: group by trip_id, and then
        #  split the data to batches of groups (trips). For each group we will
        #  apply the 'create direct edges' using a pool.
        # TODO: Consider using `progress_apply` from tqdm library to present
        #  the progress to the user.
        raw_nodes_df[['node_id', 'trip_id', 'stop_sequence', 'arrival',
                  'departure']].groupby('trip_id').apply(
            self.create_direct_edges_for_trip)

    def create_direct_edges_for_trip(self, raw_nodes_by_trip):
        # TODO: Save node_id instead of index
        for index, node in raw_nodes_by_trip.iterrows():
            stop_seq = node['stop_sequence']
            # For the same trip we want to take the next node
            next_node = raw_nodes_by_trip[
                stop_seq + 1 == raw_nodes_by_trip['stop_sequence']]
            if next_node.shape[0] == 0:
                # This is the last node of the current trip, no outgoing edge
                continue
            assert next_node.shape[0] == 1

            w = ((next_node['departure'] - node['arrival'])).values[
                    0] / np.timedelta64(1, 's')
            d_edge = (node['node_id'], next_node['node_id'].values[0], w)
            self._direct_edges.add(d_edge)

    @staticmethod
    def construct_edges(self, direct_edges, transfer_edges={}):
        return [(str(u), str(v), w) for u, v, w in
                set(direct_edges).union(set(transfer_edges))]

    def construct_nodes(self):
        # TODO: Make this function construct the actual nodes from the raw data.

        # For now, nodes are created based on all edges, so the graph edges
        # should be initialized first.
        if self._edges is None:
            raise Exception("Graph edges must be initialized before calling "
                            "construct_nodes")

        nodes = [(s, d) for s, d, _ in self._edges]
        return list(set([item for sublist in nodes for item in sublist]))

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

    def get_shortest_paths(self, sources, save_to_files=True, debug=False,
                           target=None):
        if not debug:
            mode = igraph.IN if self._reversed else igraph.OUT
            sp = self._graph.shortest_paths_dijkstra(source=sources,
                                                     target=self._nodes,
                                                     weights='weight',
                                                     mode=mode)

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

            reachable_df = reachable_df.merge(
                self._nodes_df, left_on='target', right_on='node_id').drop(
                ['target', 'node_id', 'departure'], axis=1)
            reachable_df = reachable_df.merge(
                self._nodes_df, left_on='source', right_on='node_id',
                suffixes=('_target', '_source')).drop(
                ['source', 'node_id', 'arrival_source'], axis=1)
            reachable_df = reachable_df.sort_values('time_sec').groupby(
                ['stop_id_source', 'stop_id_target'], as_index=False).first()

            if save_to_files:
                paths_type = 'sa' if self._reversed else 'aa'  # Service Area or Access Area
                filename = OUTPUT_PATH + paths_type + '_' + str(
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


VALIDATION = True
V_PATH = 'validation/test1/' if VALIDATION else ''

START_NODES_PATH = '../output_data/' + V_PATH + 'morning_start_nodes.pkl'
TARGET_NODES_PATH = '../output_data/' + V_PATH + 'target_nodes.pkl'
ALL_NODES_PATH = '../output_data/' + V_PATH + 'morning_nodes.pkl'
DIRECT_EDGES_PATH = '../output_data/' + V_PATH + 'morning_direct_edges.pkl'
# DIRECT_EDGES_PATH = '../output_data/single_trip_direct_edges.pkl'
TRANSFER_EDGES_PATH = '../output_data/' + V_PATH + 'morning_transfer_edges.pkl'
OUTPUT_PATH = '../output_data/' + V_PATH
DEBUG = False

# Indicates whether we should compute for a service area instead of access area
SERVICE = False

if __name__ == '__main__':
    direct_edges = pd.read_pickle(DIRECT_EDGES_PATH)
    transfer_edges = pd.read_pickle(TRANSFER_EDGES_PATH)
    all_nodes_df = pd.read_pickle(ALL_NODES_PATH)[
        ['node_id', 'stop_id', 'stop_lon', 'stop_lat', 'departure', 'arrival']]
    gtfs_graph = GtfsGraph(direct_edges, transfer_edges, all_nodes_df,
                           reversed_graph=SERVICE)
    gtfs_graph.construct_graph()
    print('Finished constructing the graph')

    # nodes = []
    if SERVICE:
        with open(TARGET_NODES_PATH, 'rb') as f:
            nodes = pickle.load(f)
            graph_nodes = set(gtfs_graph.get_nodes())
            nodes = [n for n in nodes if n in graph_nodes]
            sleep(2)
    else:
        with open(START_NODES_PATH, 'rb') as f:
            nodes = pickle.load(f)
            graph_nodes = set(gtfs_graph.get_nodes())
            nodes = [n for n in nodes if n in graph_nodes]

    num_of_sources = len(nodes)
    batch_size = 200
    pool_input = batches(nodes, batch_size)

    start = timer()
    p = Pool()
    p.map(gtfs_graph.get_shortest_paths, pool_input)
    p.close()
    p.join()
    end = timer()
    pool_tot_time = end - start
    time_for_source = (end - start) / num_of_sources
    print("Pool took: ", pool_tot_time, " seconds, so ", time_for_source,
          " per source")
