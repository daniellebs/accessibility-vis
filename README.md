# accessibility-vis  
## Background
For general background and details on the graphs structure please see [this document](https://drive.google.com/file/d/1-iNNNy8CAmvFpwA3jJ4Zp7h7hPhs3Y9g/view?usp=sharing). Relevant details for this repository can be found under "Chapter 3 - Research Methodology".
<!---
### Public Transportation 
#### Data: GTFS
#### Graph
### Roads 
#### Data
#### Graphs
-->
## Directories  
  
### data_processing  
Computing travel times and general accessibility.  
#### Files
##### gtfs_graph.py
The main code which computes the shortest paths between public transit stops. 
Input: weighted edges between nodes in the public transit network. See "Background" for more details on the network structure.
Output: multiple files with travel times between stops.
Implementation: Computations are multithreaded, and use the igraph library. 
##### roads_graph.py
Similar to gtfs_graph.py, adjusted to the roads graph format (each nodes directly correlates to a road junction).
##### gtfs_graph_constructor.py
This is an attempt to encapsulate the entire public transit graph construction into a single python script. Prefer using the relevant jupyter notebook for this purpose. See the next section for more details. 
#### notebooks directory
This directory includes various Jupyter notebooks that either explore the data, or more importantly construct the graph nodes and edges that are the input for the gtfs_graph.py and roads_graph.py scripts. For more details on the graphs structure see the "Backgound" chapter.
  
#### testing directory
The testing directory is divided to two frameworks that were testes:
1. Neo4j
2. igraph Python library

It's been decided to use the igraph library due to its simplicity and high performance. 
To test the scalability of both frameworks, each contains a grid graph generator. 
  
---  
  
## Non-Git Directories
These directories contain either heavy or sensitive data that cannot be uploaded to Github. 
To get access to the data please contact the owner at bensdani@post.bgu.ac.il.   
  
### input_data  
The data contains GTFS (General Transit Feed Specificatio) feeds published by the israeli ministry of transportation (MOT), and other input data for accessibility computations.   
  
The most recent GTFS data can by downloaded from ftp://199.203.58.18/.  

The input data also includes the entire Israel roads network. 
  
### output_data  
Results of accessibility computations.  
  
### vis  
Visualization of accessibility ratios.

