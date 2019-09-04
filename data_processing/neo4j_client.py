import argparse
import json
import logging
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Constructs and handles Neo4j graph based on given '
                'nodes and links.')
parser.add_argument('--credentials', metavar='C', nargs=1, type=str,
                    help='Path to credentials JSON file')

args = parser.parse_args()


class UnimplementedException(BaseException):
    pass


class Neo4jClient(object):
    def __init__(self, uri, user, password, log_level=logging.INFO):
        # logging.basicConfig(filename="neo4j.log",
        #                     format='%(asctime)s %(message)s',
        #                     filemode='w')
        self._logger = logging.getLogger()
        self._logger.setLevel(log_level)
        self._logger.info(f'Connecting to {uri}')
        self._driver = GraphDatabase.driver(uri,
                                            auth=basic_auth(user, password))

    def close(self):
        self._driver.close()

    def add_nodes(self, nodes_df, *argv):
        """Adds given nodes to the connected graph.

        :param nodes_df: pandas dataframe containing list of nodes and their
                         properties.
        :param argv: arguments representing the nodes' properties names.
        :return: None
        """
        with self._driver.session() as session:
            for index, row in tqdm(nodes_df.iterrows(), total=len(nodes_df),
                                   unit='nodes'):
                arguments = {}
                for arg in argv:
                    arguments[arg] = row[arg]
                session.write_transaction(self._add_node, **arguments)
            session.write_transaction(self._create_index)

    @staticmethod
    def _add_node(tx, **kwargs):
        """Adds a single node to the graph using the provided transaction.

        Must be overridden.
        :param tx: transaction to run.
        :param kwargs: keyword parameters for the node properties.
        :return: None
        """
        raise UnimplementedException()

    @staticmethod
    def _create_index(tx):
        """Creates a new index using the provided transaction.

        Must be overridden.
        :param tx: transaction to run.
        :return: None
        """
        raise UnimplementedException()

    def add_edges(self, edges_df, *argv):
        """Adds given edges to the connected graph.

        :param edges_df: pandas dataframe containing list of edges.
        :param argv: arguments representing the nodes' properties names.
        :return: None
        """
        with self._driver.session() as session:
            for index, row in tqdm(edges_df.iterrows(), total=len(edges_df),
                                   unit='edges'):
                arguments = {}
                for arg in argv:
                    arguments[arg] = row[arg]
                session.write_transaction(self._add_edge, **arguments)

    @staticmethod
    def _add_edge(tx, **kwargs):
        """Adds a single edge to the graph using the provided transaction.

        Must be overridden.
        :param tx: transaction to run.
        :param kwargs: keyword parameters for the node properties to match.
        :return: None
        """
        raise UnimplementedException()

    def get_single_source_shortest_paths(self, id_field, id_value):
        def sssp(tx):
            query = f"MATCH (n:Node {{{id_field}:'{id_value}'}}) " \
                f"CALL algo.shortestPath.deltaStepping.stream(n, 'time', 3.0) " \
                f"YIELD nodeId, distance " \
                f"RETURN algo.asNode(nodeId).name AS destination, distance"
            result = tx.run(query)
            return pd.DataFrame([r.values() for r in result],
                                columns=result.keys())

        with self._driver.session() as session:
            # TODO(danielle): save results
            return session.write_transaction(sssp)


if __name__ == '__main__':
    # Verify the connection succeeds
    with open(args.credentials[0]) as f:
        credentials = json.load(f)
    h = Neo4jClient(credentials['neo4j']['uri'], credentials['neo4j']['user'],
                    credentials['neo4j']['password'])
    h.close()
