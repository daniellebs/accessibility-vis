import unittest
from data_processing.gtfs_graph import GtfsGraph
import pandas as pd
import numpy as np


def time(t):
    return np.datetime64('2020-01-01T' + t)


class GtfsGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gtfs_graph = GtfsGraph({})

    def test_create_direct_edges(self):
        # Construct simple nodes with a single line
        raw_nodes = [
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
        # Construct simple nodes with a single line
        raw_nodes = [
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


if __name__ == '__main__':
    unittest.main()
