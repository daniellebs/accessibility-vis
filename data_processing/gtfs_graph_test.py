import unittest
from data_processing.gtfs_graph import GtfsGraph
import pandas as pd
import numpy as np


def time(t):
    return np.datetime64('2020-01-01T' + t)


#    OO O o o o...      ______________________ _________________
#   O     ____          |                    | |               |
#  ][_n_i_| (   ooo___  |                    | |               |
# (__________|_[______]_|____________________|_|_______________|
#   0--0--0      0  0      0       0     0        0        0

class GtfsGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gtfs_graph = GtfsGraph({})

    def test_create_direct_edges(self):
        # Construct simple nodes with a single trip.
        raw_nodes = [
            # trip, sequence, arrival, departure, node
            [1, 1, time('08:00'), time('08:00'), 1],
            [1, 2, time('08:05'), time('08:05'), 2],
            [1, 3, time('08:06'), time('08:06'), 3]]
        raw_nodes_df = pd.DataFrame(raw_nodes,
                                    columns=['trip_id',
                                             'stop_sequence',
                                             'arrival',
                                             'departure',
                                             'node_id'])

        # Create direct edges and verify result.
        self.gtfs_graph.create_direct_edges(raw_nodes_df)
        self.assertEqual({(1, 2, 5*60), (2, 3, 1*60)},
                         self.gtfs_graph.get_direct_edges())

    def test_create_direct_edges_two_trips(self):
        # Construct simple nodes with two trips.
        raw_nodes = [
            # trip, sequence, arrival, departure, node
            [1, 1, time('08:00'), time('08:00'), 1],
            [3, 1, time('09:10'), time('09:10'), 2],
            [1, 2, time('08:05'), time('08:05'), 3],
            [3, 2, time('09:20'), time('09:20'), 4],
            [1, 3, time('08:06'), time('08:06'), 5]]
        raw_nodes_df = pd.DataFrame(raw_nodes,
                                    columns=['trip_id',
                                             'stop_sequence',
                                             'arrival',
                                             'departure',
                                             'node_id'])

        # Create direct edges and verify result.
        self.gtfs_graph.create_direct_edges(raw_nodes_df)
        self.assertEqual({(1, 3, 5*60),
                          (3, 5, 1*60),
                          (2, 4, 10*60)},
                         self.gtfs_graph.get_direct_edges())

    def test_create_transfer_edges_same_stop_transfer(self):
        # Construct simple nodes with two trips.
        #                    ┌─────────────┐
        #                    │   Stop:1    │
        #                    │ LineA: 8:00 │
        #                    └──────┬──────┘
        #                           │
        # ┌─────────────┐    ┌──────▼──────┐    ┌─────────────┐
        # │   Stop:5    │    │   Stop:2    │    │   Stop:5    │
        # │ LineB: 8:02 ├───►│ LineA: 8:10 ├───►│ LineB: 8:16 │
        # │             │    │ LineB: 8:13 │    │             │
        # └─────────────┘    └──────┬──────┘    └─────────────┘
        #                           │
        #                    ┌──────▼──────┐
        #                    │   Stop:3    │
        #                    │ LineA: 8:15 │
        #                    └──────┬──────┘
        #                           │
        #                    ┌──────▼──────┐
        #                    │   Stop:4    │
        #                    │ LineA: 8:20 │
        #                    └─────────────┘
        raw_nodes = [
            # stop, trip, route, sequence, arrival, departure, node
            [1, 1, 1, 1, time('08:00'), time('08:00'), 1],
            [2, 1, 1, 2, time('08:10'), time('08:10'), 2],
            [3, 1, 1, 3, time('08:15'), time('08:15'), 3],
            [4, 1, 1, 4, time('08:20'), time('08:20'), 4],
            [5, 2, 2, 1, time('08:02'), time('08:02'), 5],
            [2, 2, 2, 2, time('08:13'), time('08:13'), 6],
            [6, 2, 2, 3, time('08:16'), time('08:16'), 7]
        ]
        raw_nodes_df = pd.DataFrame(raw_nodes,
                                    columns=['stop_id',
                                             'trip_id',
                                             'route_id',
                                             'stop_sequence',
                                             'arrival',
                                             'departure',
                                             'node_id'])

        # Construct simple stops distances table.
        stops_distances = [
            # StopA, StopB, Distance
            [1, 1, 0],
            [2, 2, 0],
            [3, 3, 0],
            [4, 4, 0],
            [5, 5, 0],
            [6, 6, 0]
        ]
        stops_distances_df = pd.DataFrame(stops_distances,
                                          columns=['from_stop_id',
                                                   'to_stop_id',
                                                   'dist'])

        # Create transfer edges and verify result.
        self.gtfs_graph.create_transfer_edges(raw_nodes_df, stops_distances_df,
                                              100)
        self.assertEqual({(2, 6, 3*60)},
                         self.gtfs_graph.get_transfer_edges())

    # TODO: Add tests for transfer between stops, non-possible transfers etc.

    def test_create_transfer_edges_walk_to_another_stop(self):
        # ┌─────────────┐           ┌─────────────┐
        # │   Stop:1    │           │   Stop:4    │
        # │ LineA: 8:00 │           │ LineB: 8:02 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐    50m    ┌──────▼──────┐
        # │   Stop:2    │xxxxxxxxxxx│   Stop:5    │
        # │ LineA: 8:05 │           │ LineB: 8:08 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐           ┌──────▼──────┐
        # │   Stop:3    │           │   Stop:6    │
        # │ LineA: 8:10 │           │ LineB: 8:15 │
        # └─────────────┘           └─────────────┘
        raw_nodes = [
            # stop, trip, route, sequence, arrival, departure, node
            [1, 1, 1, 1, time('08:00'), time('08:00'), 1],
            [2, 1, 1, 2, time('08:05'), time('08:05'), 2],
            [3, 1, 1, 3, time('08:10'), time('08:10'), 3],
            [4, 2, 2, 1, time('08:02'), time('08:02'), 4],
            [5, 2, 2, 2, time('08:08'), time('08:08'), 5],
            [6, 2, 2, 3, time('08:15'), time('08:15'), 6]
        ]
        raw_nodes_df = pd.DataFrame(raw_nodes,
                                    columns=['stop_id',
                                             'trip_id',
                                             'route_id',
                                             'stop_sequence',
                                             'arrival',
                                             'departure',
                                             'node_id'])

        # Construct simple stops distances table.
        stops_distances = [
            # StopA, StopB, Distance
            [1, 1, 0],
            [2, 2, 0],
            [3, 3, 0],
            [4, 4, 0],
            [5, 5, 0],
            [6, 6, 0],
            [2, 5, 50]
        ]
        stops_distances_df = pd.DataFrame(stops_distances,
                                          columns=['from_stop_id',
                                                   'to_stop_id',
                                                   'dist'])

        # Create transfer edges and verify result.
        self.gtfs_graph.create_transfer_edges(raw_nodes_df, stops_distances_df,
                                              100)
        self.assertEqual({(2, 5, 3 * 60)},
                         self.gtfs_graph.get_transfer_edges())

    def test_create_transfer_too_far(self):
        # ┌─────────────┐           ┌─────────────┐
        # │   Stop:1    │           │   Stop:4    │
        # │ LineA: 8:00 │           │ LineB: 8:02 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐   200m    ┌──────▼──────┐
        # │   Stop:2    │xxxxxxxxxxx│   Stop:5    │
        # │ LineA: 8:05 │           │ LineB: 8:10 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐           ┌──────▼──────┐
        # │   Stop:3    │           │   Stop:6    │
        # │ LineA: 8:10 │           │ LineB: 8:15 │
        # └─────────────┘           └─────────────┘
        raw_nodes = [
            # stop, trip, route, sequence, arrival, departure, node
            [1, 1, 1, 1, time('08:00'), time('08:00'), 1],
            [2, 1, 1, 2, time('08:05'), time('08:05'), 2],
            [3, 1, 1, 3, time('08:10'), time('08:10'), 3],
            [4, 2, 2, 1, time('08:02'), time('08:02'), 4],
            [5, 2, 2, 2, time('08:10'), time('08:10'), 5],
            [6, 2, 2, 3, time('08:15'), time('08:15'), 6]
        ]
        raw_nodes_df = pd.DataFrame(raw_nodes,
                                    columns=['stop_id',
                                             'trip_id',
                                             'route_id',
                                             'stop_sequence',
                                             'arrival',
                                             'departure',
                                             'node_id'])

        # Construct simple stops distances table.
        stops_distances = [
            # StopA, StopB, Distance
            [1, 1, 0],
            [2, 2, 0],
            [3, 3, 0],
            [4, 4, 0],
            [5, 5, 0],
            [6, 6, 0],
            [2, 5, 200]
        ]
        stops_distances_df = pd.DataFrame(stops_distances,
                                          columns=['from_stop_id',
                                                   'to_stop_id',
                                                   'dist'])

        # Create transfer edges and verify result: no valid transfer.
        self.gtfs_graph.create_transfer_edges(raw_nodes_df, stops_distances_df,
                                              100)
        self.assertEqual(set(),
                         self.gtfs_graph.get_transfer_edges())

    def test_create_transfer_takes_too_long(self):
        # ┌─────────────┐           ┌─────────────┐
        # │   Stop:1    │           │   Stop:4    │
        # │ LineA: 8:00 │           │ LineB: 8:10 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐    50m    ┌──────▼──────┐
        # │   Stop:2    │xxxxxxxxxxx│   Stop:5    │
        # │ LineA: 8:05 │           │ LineB: 8:21 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐           ┌──────▼──────┐
        # │   Stop:3    │           │   Stop:6    │
        # │ LineA: 8:10 │           │ LineB: 8:22 │
        # └─────────────┘           └─────────────┘
        raw_nodes = [
            # stop, trip, route, sequence, arrival, departure, node
            [1, 1, 1, 1, time('08:00'), time('08:00'), 1],
            [2, 1, 1, 2, time('08:05'), time('08:05'), 2],
            [3, 1, 1, 3, time('08:10'), time('08:10'), 3],
            [4, 2, 2, 1, time('08:02'), time('08:10'), 4],
            [5, 2, 2, 2, time('08:10'), time('08:21'), 5],
            [6, 2, 2, 3, time('08:15'), time('08:22'), 6]
        ]
        raw_nodes_df = pd.DataFrame(raw_nodes,
                                    columns=['stop_id',
                                             'trip_id',
                                             'route_id',
                                             'stop_sequence',
                                             'arrival',
                                             'departure',
                                             'node_id'])

        # Construct simple stops distances table.
        stops_distances = [
            # StopA, StopB, Distance
            [1, 1, 0],
            [2, 2, 0],
            [3, 3, 0],
            [4, 4, 0],
            [5, 5, 0],
            [6, 6, 0],
            [2, 5, 200]
        ]
        stops_distances_df = pd.DataFrame(stops_distances,
                                          columns=['from_stop_id',
                                                   'to_stop_id',
                                                   'dist'])

        # Create transfer edges and verify result: no valid transfer.
        self.gtfs_graph.create_transfer_edges(raw_nodes_df, stops_distances_df,
                                              100)
        self.assertEqual(set(),
                         self.gtfs_graph.get_transfer_edges())

    def test_create_transfer_dont_make_it_on_time(self):
        # The passengers can't walk from Stop 2 to Stop 5 in 1 minute, thus
        # can't make this transfer on time for board LineB.
        # ┌─────────────┐           ┌─────────────┐
        # │   Stop:1    │           │   Stop:4    │
        # │ LineA: 8:00 │           │ LineB: 8:01 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐    90m    ┌──────▼──────┐
        # │   Stop:2    │xxxxxxxxxxx│   Stop:5    │
        # │ LineA: 8:05 │           │ LineB: 8:06 │
        # └──────┬──────┘           └──────┬──────┘
        #        │                         │
        # ┌──────▼──────┐           ┌──────▼──────┐
        # │   Stop:3    │           │   Stop:6    │
        # │ LineA: 8:10 │           │ LineB: 8:10 │
        # └─────────────┘           └─────────────┘
        raw_nodes = [
            # stop, trip, route, sequence, arrival, departure, node
            [1, 1, 1, 1, time('08:00'), time('08:00'), 1],
            [2, 1, 1, 2, time('08:05'), time('08:05'), 2],
            [3, 1, 1, 3, time('08:10'), time('08:10'), 3],
            [4, 2, 2, 1, time('08:02'), time('08:01'), 4],
            [5, 2, 2, 2, time('08:10'), time('08:06'), 5],
            [6, 2, 2, 3, time('08:15'), time('08:10'), 6]
        ]
        raw_nodes_df = pd.DataFrame(raw_nodes,
                                    columns=['stop_id',
                                             'trip_id',
                                             'route_id',
                                             'stop_sequence',
                                             'arrival',
                                             'departure',
                                             'node_id'])

        # Construct simple stops distances table.
        stops_distances = [
            # StopA, StopB, Distance
            [1, 1, 0],
            [2, 2, 0],
            [3, 3, 0],
            [4, 4, 0],
            [5, 5, 0],
            [6, 6, 0],
            [2, 5, 90]  # 90m, based on average walk speed of 1m/s, meaning
                        # it'll take 90 seconds to walk from stop 2 to stop 5.
        ]
        stops_distances_df = pd.DataFrame(stops_distances,
                                          columns=['from_stop_id',
                                                   'to_stop_id',
                                                   'dist'])

        # Create transfer edges and verify result: no valid transfer.
        self.gtfs_graph.create_transfer_edges(raw_nodes_df, stops_distances_df,
                                              100)
        self.assertEqual(set(),
                         self.gtfs_graph.get_transfer_edges())


if __name__ == '__main__':
    unittest.main()
