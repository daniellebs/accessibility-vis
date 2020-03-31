from multiprocessing import Pool
from timeit import default_timer as timer

import igraph
import pandas as pd


class RoadsGraph:
    def __init__(self, edges, reversed_graph=False):
        self._edges = [(str(u), str(v), w) for u, v, w in edges]
        nodes = [(s, d) for s, d, _ in self._edges]
        self._nodes = list(set([item for sublist in nodes for item in sublist]))
        self._graph = igraph.Graph()
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


SERVICE = False
S_PATH = 'sa' if SERVICE else ''
VALIDATION = False
V_PATH = 'validation/roads_network/' if VALIDATION else ''

START_NODES_PATH = '../output_data/' + V_PATH + 'morning_start_nodes.pkl'
TARGET_NODES_PATH = '../output_data/' + V_PATH + 'target_nodes.pkl'
EDGES_PATH = '../output_data/' + V_PATH + 'roads_graph_weighted_edges.pkl'
OUTPUT_PATH = '../output_data/' + V_PATH + 'roads/' + S_PATH
DEBUG = False

if __name__ == '__main__':
    edges = pd.read_pickle(EDGES_PATH)
    roads_graph = (edges, SERVICE)
    roads_graph.construct_graph()
    print('Finished constructing the graph')

    nodes = roads_graph.get_nodes()
    num_of_sources = len(nodes)
    batch_size = 200
    pool_input = batches(nodes, batch_size)

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