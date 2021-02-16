from multiprocessing import Pool
from timeit import default_timer as timer

import igraph
import pandas as pd
import pickle
import datetime

MINUTE = 60

class RoadsGraph:
    def __init__(self, edges, node_attributes, reversed_graph=False):
        self._edges = [(str(u), str(v), w) for u, v, w in edges]
        nodes = [(s, d) for s, d, _ in self._edges]
        self._nodes = list(set([item for sublist in nodes for item in sublist]))
        self._graph = igraph.Graph()
        self._nodes_attributes = node_attributes
        self._reversed = reversed_graph

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



            # ==================================================================
            # This code is relevant if we want to save for each node which other
            # nodes are reachable within max_len_sec

            # reachable = {'source': [], 'target': [], 'time_sec': []}
            # for i, source in enumerate(sources):
            #     # Verify all results and correct length
            #     assert len(self._nodes) == len(sp[i])
            #     dists = sp[i]
            #
            #     # Go through all targets and save only reachable nodes
            #     for target_i, d in enumerate(dists):
            #         if d <= max_len_sec:
            #             reachable['source'].append(int(source))
            #             reachable['target'].append(int(self._nodes[target_i]))
            #             reachable['time_sec'].append(d)
            #
            # reachable_df = pd.DataFrame.from_dict(reachable)
            # del reachable  # Delete reachable dict from memory
            #
            # if save_to_files:
            #     paths_type = 'sa' if self._reversed else 'aa'  # Service Area or Access Area
            #     filename = OUTPUT_PATH + '/' + paths_type + '/' + paths_type + \
            #                '_' + str(sources[0]) + '-' + \
            #                str(sources[-1]) + '.pkl'
            #     reachable_df.to_pickle(filename)
            # ==================================================================

            # "areas" can be access_areas or service_areas, depending on the
            # reversed variable.
            areas = dict()
            for i, source in enumerate(sources):
                # Verify all results and correct length
                assert len(self._nodes) == len(sp[i])
                times_to_targets = sp[i]

                source_int = int(source)
                areas[source_int] = 0
                # areas[source_int] = dict()

                # Go through all targets and save only reachable nodes
                for target_i, time_to_target in enumerate(times_to_targets):
                    target_attr = self._nodes_attributes[self._nodes[target_i]]
                    # T = max_len_sec
                    # while T >= time_to_target and T >= 15*MINUTE:
                    #     if T not in areas[source_int]:
                    #         areas[source_int][T] = 0
                    #     areas[source_int][T] += target_attr
                    #     T -= 2*MINUTE
                    if time_to_target < MAX_TIME_SEC:
                        areas[source_int] += target_attr

            if save_to_files:
                paths_type = 'sa' if self._reversed else 'aa'  # Service Area or Access Area
                filename = OUTPUT_PATH + '/' + paths_type + '/' + paths_type + \
                           '_' + str(sources[0]) + '-' + \
                           str(sources[-1]) + '.pkl'
                pickle.dump(areas, open(filename, 'wb'))

        if debug:
            print(f'Trying to get a shortest path from {sources} to {target}')
            assert target is not None, 'In debug mode the target must be set'
            sp = self._graph.get_shortest_paths_dijkstra(sources, to=target,
                                                         weights='weight',
                                                         mode=igraph.OUT)
            print('======================')
            print(f'Path to node {sp[-1]} is {sp}')


def batches(l, n):
    """Yield successive n-sized batches from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


TEST = False
T_PATH = 'test/haifa_krayot/' if TEST else ''
USE_START_NODES = True
START_NODES_PATH = '../output_data/roads/test/center_tlv/center_tlv_nodes.pkl'

EDGES_PATH = '../output_data/roads/' + T_PATH + 'roads_graph_weighted_edges.pkl'
ATTRIBUTES_PATH = '../output_data/roads/' + T_PATH + 'roads_graph_node_attributes.pkl'

OUTPUT_PATH = '../output_data/roads/' + T_PATH
DEBUG = False

MAX_TIME_SEC = 45 * MINUTE

# Should we compute for a service ares instead of access area?
SERVICE = False

if __name__ == '__main__':
    edges = pd.read_pickle(EDGES_PATH)
    node_attributes = pickle.load(open(ATTRIBUTES_PATH, 'rb'))
    roads_graph = RoadsGraph(edges, node_attributes, SERVICE)
    roads_graph.construct_graph()
    print('Finished constructing the graph')

    # nodes = list()
    if USE_START_NODES:
        nodes = pickle.load(open(START_NODES_PATH, 'rb'))
    else:
        nodes = roads_graph.get_nodes()
    num_of_sources = len(nodes)
    batch_size = 50
    pool_input = batches(nodes, batch_size)

    print(f'Started at time {datetime.datetime.now()}')

    start = timer()
    p = Pool()
    p.map(roads_graph.get_shortest_paths, pool_input)
    p.close()
    p.join()
    end = timer()
    pool_tot_time = end - start
    time_for_source = (end - start) / num_of_sources
    print("Pool took: ", pool_tot_time, " seconds, so ", time_for_source,
          " per source")

    # ============ Performance testing ============
    # nodes = sorted(roads_graph.get_nodes()[:500])  # Lexicographically sorted and not numerically
    # num_of_sources = len(nodes)
    # batch_size = 50
    # pool_input = batches(nodes, batch_size)
    #
    # start = timer()
    # p = Pool()
    # p.map(roads_graph.get_shortest_paths, pool_input)
    # p.close()
    # p.join()
    # end = timer()
    # pool_tot_time = end - start
    # time_for_source = (end - start) / num_of_sources
    # print("Pool took: ", pool_tot_time, " seconds, so ", time_for_source,
    #       " per source")
