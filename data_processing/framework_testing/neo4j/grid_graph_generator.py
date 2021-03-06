from data_processing.framework_testing.neo4j.neo4j_client import Neo4jClient, args
import json
import numpy as np
import pandas as pd
from timeit import default_timer as timer

class GraphGenerator(Neo4jClient):
    def __init__(self, uri, user, password, number_of_rows, number_of_columns,
                 edge_probability=0.5, weight_mean=1, weight_std=0.2,
                 dry_run=False):
        super(GraphGenerator, self).__init__(uri, user, password)
        self._number_of_rows = number_of_rows
        self._number_of_cols = number_of_columns
        self._edge_prob = edge_probability
        self._weight_mean = weight_mean
        self._weight_std = weight_std
        self._nodes = None
        self._edges = []
        self._dry_run = dry_run

    def bernoulli_trial(self):
        return np.random.random() > self._edge_prob

    def weight(self):
        # return np.random.normal(self._weight_mean, self._weight_std)
        return np.random.uniform(0.0, 3.0)

    def generate_planar_grid_graph(self):
        # Create cartesian product
        rows = range(self._number_of_rows)
        cols = range(self._number_of_cols)
        self._nodes = [(r, c) for r in rows for c in cols]
        self._logger.info(f'Generated {len(self._nodes)} nodes.')
        for i, j in self._nodes:
            # Can also be done by iterating from (i-1,j-1) to (i+1,j+1).
            if i > 0 and j > 0 and self.bernoulli_trial():
                self._edges.append((i, j, i - 1, j - 1, self.weight()))
            if i > 0 and self.bernoulli_trial():
                self._edges.append((i, j, i - 1, j, self.weight()))
            if i > 0 and j < self._number_of_cols - 1 \
                    and self.bernoulli_trial():
                self._edges.append((i, j, i - 1, j + 1, self.weight()))
            if j < self._number_of_cols - 1 and self.bernoulli_trial():
                self._edges.append((i, j, i, j + 1, self.weight()))
            if i < self._number_of_rows - 1 and j < self._number_of_cols - 1 \
                    and self.bernoulli_trial():
                self._edges.append((i, j, i + 1, j + 1, self.weight()))
            if i < self._number_of_rows - 1 and self.bernoulli_trial():
                self._edges.append((i, j, i + 1, j, self.weight()))
            if i < self._number_of_rows - 1 and j > 0 \
                    and self.bernoulli_trial():
                self._edges.append((i, j, i + 1, j - 1, self.weight()))
            if j > 0:
                self._edges.append((i, j, i, j - 1, self.weight()))

        # self._nodes_to_index = {k: v for v, k in enumerate(self._nodes)}
        #
        # for i, j in self._nodes:
        #     curr_index = self._nodes_to_index[(i, j)]
        #     # Can also be done by iterating from (i-1,j-1) to (i+1,j+1).
        #     if i > 0 and j > 0 and self.bernoulli_trial():
        #         self._edges.append((curr_index,
        #                             self._nodes_to_index[(i - 1, j - 1)],
        #                             self.weight()))
        #     if i > 0 and self.bernoulli_trial():
        #         self._edges.append((
        #             curr_index, self._nodes_to_index[(i - 1, j)],
        #             self.weight()))
        #     if i > 0 and j < self._number_of_cols - 1 \
        #             and self.bernoulli_trial():
        #         self._edges.append((curr_index,
        #                             self._nodes_to_index[(i - 1, j + 1)],
        #                             self.weight()))
        #     if j < self._number_of_cols - 1 and self.bernoulli_trial():
        #         self._edges.append((
        #             curr_index, self._nodes_to_index[(i, j + 1)],
        #             self.weight()))
        #     if i < self._number_of_rows - 1 and j < self._number_of_cols - 1 \
        #             and self.bernoulli_trial():
        #         self._edges.append((curr_index,
        #                             self._nodes_to_index[(i + 1, j + 1)],
        #                             self.weight()))
        #     if i < self._number_of_rows - 1 and self.bernoulli_trial():
        #         self._edges.append((
        #             curr_index, self._nodes_to_index[(i + 1, j)],
        #             self.weight()))
        #     if i < self._number_of_rows - 1 and j > 0 \
        #             and self.bernoulli_trial():
        #         self._edges.append((curr_index,
        #                             self._nodes_to_index[(i + 1, j - 1)],
        #                             self.weight()))
        #     if j > 0:
        #         self._edges.append((
        #             curr_index, self._nodes_to_index[(i, j - 1)],
        #             self.weight()))

        self._logger.info(f'Generated {len(self._edges)} edges.')

        if not self._dry_run:
            # Create index constraint
            def index_constraint(tx):
                query = "CREATE CONSTRAINT ON (n:Node) ASSERT (n.i, n.j) IS NODE KEY;"
                tx.run(query)

            with self._driver.session() as session:
                session.write_transaction(index_constraint)

            # Add nodes to graph
            # nodes_columns = ['i', 'j']
            # self.add_nodes(pd.DataFrame(self._nodes, columns=nodes_columns),
            #                *nodes_columns)

            # Add edges to graph
            edges_columns = ['from_i', 'from_j', 'to_i', 'to_j', 'w']
            self.add_edges(pd.DataFrame(self._edges, columns=edges_columns),
                           *edges_columns)
            # edges_columns = ['a', 'b', 'w']
            # self.add_edges(pd.DataFrame(self._edges, columns=edges_columns),
            #                *edges_columns)

    def get_number_of_nodes(self):
        return len(self._nodes)

    def get_number_of_edges(self):
        return len(self._edges)

    @staticmethod
    def _add_node(tx, **kwargs):
        """Override"""
        i = kwargs['i']
        j = kwargs['j']
        query = f"CREATE (a:Node {{name:'{str((i, j))}'}}) SET " \
            f"a.i = {i} SET a.j = {j}"
        tx.run(query)

    @staticmethod
    def _create_index(tx):
        """Override"""
        # tx.run("CREATE INDEX ON :Node(i, j)")
        # tx.run("CREATE INDEX ON :Node(n)")
        pass

    @staticmethod
    def _add_edge(tx, **kwargs):
        """Override"""
        # query = f"MATCH (a:Node),(b:Node) WHERE " \
        #     f"a.i = {int(kwargs['from_i'])} AND " \
        #     f"a.j = {int(kwargs['from_j'])} AND " \
        #     f"b.i = {int(kwargs['to_i'])} AND " \
        #     f"b.j = {int(kwargs['to_j'])} CREATE " \
        #     f"(a)-[:CONNECTS {{time: {kwargs['w']}}}]->(b)"
        # query = f"MERGE (a:Node {{n: {int(kwargs['a'])}}}) " \
        #     f"MERGE (b:Node {{n: {int(kwargs['b'])}}}) " \
        #     f"MERGE (a)-[:CONNECTS {{time: {kwargs['w']}}}]->(b)"
        query = f"MERGE (a:Node {{i: {int(kwargs['from_i'])}, " \
            f"j: {int(kwargs['from_j'])}}}) " \
            f"MERGE (b:Node {{i: {int(kwargs['to_i'])}, " \
            f"j: {int(kwargs['to_j'])}}}) " \
            f"MERGE (a)-[:CONNECTS {{time: {kwargs['w']}}}]->(b)"
        tx.run(query)


if __name__ == '__main__':
    with open(args.credentials[0]) as f:
        credentials = json.load(f)
        graph_generator = GraphGenerator(credentials['neo4j']['uri'],
                                         credentials['neo4j']['user'],
                                         credentials['neo4j']['password'], 500,
                                         500, 0.5, 1, 0.4)
        # graph_generator.generate_planar_grid_graph()
        print('loading graph to memory')
        graph_generator.load_graph_to_memory()
        print('getting SSSP')
        start = timer()
        sssp = graph_generator.get_single_source_shortest_paths('i', 0, 'j', 0)
        # print(sssp)
        # pd.to_pickle(sssp, 'sssp_test.pkl')
        end = timer()
        print(f'Took {(end - start)*1000} ms to compute SSSP')
        graph_generator.close()
