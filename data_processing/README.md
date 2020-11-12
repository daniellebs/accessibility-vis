# How to use this directory

_Note: each notebook sets a data path at the beginning. This should be modified and directed to the relevant input data, as mentioned in each step._

## Public Transit
To construct a new graph:
1. Get nodes: Run PT-Accessibility-Nodes notebook on input GTFS data. Note that this should also output "start nodes" which indicate the possible nodes one can leave from (since here, each node represent an event in time, we limit the time window in wich we can start our journey through the transit graph. For example, we can take only start nodes with a departure time between 8:00 and 8:30. This can ve configured in the code.).
2. Get direct edges: Run PT-Accessibility-Direct-Edges notebook on input GTFS data and computed nodes.
3. Get transfer edges: Run PT-Accessibility-Transfer-Edges notebook on input GTFS data and computed nodes.
4. Once all files representing the graph were constructed, run gtfs_graph.py to perform actual computations. Set a Output will be a map of reachable nodes (within the defined time limit) from all nodes to all others, split into batches (multiple files). Note that a stop can appear in more than one batch (since this computes paths from nodes-to-nodes. Each node is a time event, we can have a <Line1,Departure1,**Stop1**> and a <Line2,Departure2,**Stop1**> as separate nodes that will appear in separate output batches), thus some aggregation on the output is required (heavy for all-israel transit stops). *Please note the capital-letters constats that are defined in the file. These allow to tweak the computations: access area or service area; testing or real data; data paths, etc.*

## Roads
To construct a new graph:
1. Run ConstructRoadsGraph notebook. This should output nodes, edges, and an additional "attributes" file which represents the accessibility attribute we want to compute (For example, number of potential jobs in each node).
2. Run roads_graph.py to perform actual computations, similarly to the transit computations, however here we also take into account the attributes, and output the following: the sum of all attributes for all reachable nodes within the defined time limit, split into batches (multiple files) (for example: if in 5 minutes, from node X we can reach node A with attribut=5, and node B with attribute=10, then we'll get `nodeX: 15`). *Please note the capital-letters constats that are defined in the file. These allow to tweak the computations: access area or service area; testing or real data; data paths, etc.*
