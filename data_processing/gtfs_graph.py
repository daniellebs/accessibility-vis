import igraph
from multiprocessing import Pool
import pickle
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from time import sleep
import datetime as dt
import argparse
import bisect

# Command line flags
parser = argparse.ArgumentParser()
# TODO: Add flags here
args = parser.parse_args()

#  .-------------------------------------------------------------.
# '------..-------------..----------..----------..----------..--.|
# |       \\            ||          ||          ||          ||  ||
# |        \\           ||          ||          ||          ||  ||
# |    ..   ||  _    _  ||    _   _ || _    _   ||    _    _||  ||
# |    ||   || //   //  ||   //  // ||//   //   ||   //   //|| /||
# |_.------"''----------''----------''----------''----------''--'|
#  |)|      |       |       |       |    |         |      ||==|  |
#  | |      |  _-_  |       |       |    |  .-.    |      ||==| C|
#  | |  __  |.'.-.' |   _   |   _   |    |.'.-.'.  |  __  | "__=='
#  '---------'|( )|'----------------------'|( )|'----------""
#              '-'                          '-'

# Represents a graphs based on GTFS data (public transit schedule).
# This class expects the input data to be valid. If you intend to use this class
# and are not sure about your data's validity, please perform your own
# validations beforehand.
class GtfsGraph:
    def __init__(self, direct_edges, transfer_edges={}, nodes_df=None,
                 reversed_graph=False):
        self._edges = self.construct_edges(direct_edges, transfer_edges)
        self._nodes = self.construct_nodes()
        self._graph = igraph.Graph()
        self._nodes_df = nodes_df
        self._reversed = reversed_graph
        self._direct_edges = set()
        self._transfer_edges = set()

    # Initializes the graph's direct edges with (NodeA_ID, NodeB_ID, seconds)
    # where NodeA and NodeB are consecutive nodes (stops) in the same trip.
    # 'seconds' is the number of seconds it takes to get from NodeA to NodeB.
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

            # TODO: This is using arrival at A -> departure from B, and we might
            #  be double counting here (we count both wait time in A and wait
            #  time in B, and then in the next edge from B to C we will count
            #  wait time at B again). Fix this.
            w = ((next_node['departure'] - node['arrival'])).values[
                    0] / np.timedelta64(1, 's')
            d_edge = (node['node_id'], next_node['node_id'].values[0], w)
            self._direct_edges.add(d_edge)

    def create_transfer_edges(self, raw_nodes_df: pd.DataFrame,
                              stops_distances_df: pd.DataFrame,
                              max_distance_meters: float):
        # assert stops_distances_df.columns == ['from_stop_id', 'to_stop_id',
        #                                       'dist']

        # Compute walking time between stops that are within the allowed
        # distance from one another.
        stops_dists_df = stops_distances_df[
            stops_distances_df['dist'] < max_distance_meters]
        # TODO: Extract SVG_WALK_SPEED to flag with default value of 1
        AVG_WALK_SPEED = 1  # meters per second (m/s)
        stops_dists_df['walk_time_sec'] = stops_dists_df[
                                                'dist'] / AVG_WALK_SPEED

        # Reshape the input nodes so that we have each stop ID mapped to all
        # nodes for that stop, sorted by departure time.
        stops_to_nodes = raw_nodes_df.groupby('stop_id')[[
            'node_id', 'trip_id', 'arrival', 'departure', 'route_id']].apply(
            lambda node: node.values.tolist()).to_dict()
        DEPARTURE_INDEX = 3  # 'departure' column index in each node.
        for stop, nodes in stops_to_nodes.items():
            stops_to_nodes[stop] = sorted(nodes,
                                          key=lambda x: x[DEPARTURE_INDEX])

        # Compute some stats
        total_values = 0
        max_nodes_in_stop = 0
        for n in stops_to_nodes.values():
            num_nodes = len(n)
            if num_nodes > max_nodes_in_stop:
                max_nodes_in_stop = num_nodes
            total_values += num_nodes
        print(
            f'There is a total of {total_values} nodes in the stops_to_nodes '
            f'dictionary')
        print(
            f'There is an average of {total_values / len(stops_to_nodes)} '
            f'nodes per stop, and a maximum of {max_nodes_in_stop} nodes in a '
            f'single stop.')

        # Compute transfer edges
        self.initialize_transfer_edges(stops_dists_df, stops_to_nodes)

    def initialize_transfer_edges(self, stops_dists_df, stops_to_nodes):
        ARRIVAL_INDEX = 2
        DEPARTURE_INDEX = 3
        ROUTE_ID_INDEX = 4
        MAX_WAIT_TIME = dt.timedelta(minutes=15)  # TODO: Extract to flag

        stops_to_nodes = dict(stops_to_nodes)
        edges = []
        for stop, nodes in stops_to_nodes.items():
            for start_n in nodes:
                nearby_stops_df = stops_dists_df[
                    stops_dists_df['from_stop_id'] == stop]
                # Add current stop to check transfers from the same stop
                nearby_stops_df.append(
                    {'from_stop_id': [stop], 'to_stop_id': [stop],
                     'dist': [0], 'walk_time_sec': [0]}, ignore_index=True)
                # TODO: verify this we're not staying on the same line in same
                #  direction
                for s in nearby_stops_df.iterrows():
                    nearby_stop_id = s[1]['to_stop_id']
                    if nearby_stop_id not in stops_to_nodes:
                        # Some stops don't have trips that operate all week.
                        # Some operate only on weekends. If this is such a stop
                        # we should continue to look at other stops, we won't
                        # find any nodes here.
                        continue
                    nearby_nodes = stops_to_nodes[nearby_stop_id]
                    second_line_earliest_start_time = \
                        start_n[ARRIVAL_INDEX] + \
                        dt.timedelta(seconds=s[1]['walk_time_sec'])
                    second_line_latest_start_time = \
                        second_line_earliest_start_time + MAX_WAIT_TIME
                    # Find index of first node that departs at least at
                    # second_start_time or later
                    _, _, _, departures, _ = zip(*nearby_nodes)
                    i = bisect.bisect_left(departures,
                                           second_line_earliest_start_time)
                    while (i < len(nearby_nodes) and
                           nearby_nodes[i][
                               DEPARTURE_INDEX] >= second_line_earliest_start_time and
                           nearby_nodes[i][
                               DEPARTURE_INDEX] <= second_line_latest_start_time):
                        node = nearby_nodes[i][0]
                        if node == start_n[0] or (
                                nearby_nodes[i][ROUTE_ID_INDEX] == start_n[
                            ROUTE_ID_INDEX] and s == stop):
                            # We don't wish to transfer to the same node (no
                            # self-edges).
                            # Another case we wish to avoid is transferring to
                            # the same line (route__id) in the same stop.
                            i += 1
                            continue
                        edges.append(
                            (start_n[0],
                             nearby_nodes[i][0],
                             nearby_nodes[i][DEPARTURE_INDEX] - start_n[ARRIVAL_INDEX]))
                        i += 1

        # TODO: initialize directly in the loop
        for s, t, w in edges:
            self._transfer_edges.add((s, t, w.total_seconds()))

    def get_direct_edges(self):
        return self._direct_edges

    def get_transfer_edges(self):
        return self._transfer_edges

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
