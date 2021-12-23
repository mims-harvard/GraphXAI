# Guide to Testing Scripts

## `datasets`

### `BAH`

### `gnn_accuracies`

Test GNN accuracy on ShapeGraph.

1. `test_models.py`: Tests different GNN architectures on the ShapeGraph.

### `graph_building`

1. `build_bound_graph.py`
2. `khop_on_graphs.py`
3. `plot_runtime_ul.py`
4. `test_one_bound.py`: Creates one version of a set of parameters for ShapeGraph. Shows degree distribution in addition to some basic characteristics of the graph.

usage: python3 test_one_ul.py <num_subgraphs> <prob_connection> <subgraph_size>

5. `test_ul_graph_gnn.py`
6. `test_ul_graph.py`
7. `test_ul_properties.py`: Grid tests class imbalance for parameters of ShapeGraph.
8. `test_ul_runtime.py`: Grid tests runtimes and sizes for parameters of ShapeGraph.

#### 

### `verification`
Scripts to verify synthetic graphs for accidental motifs.

1. `house_checker.py`: Generates matrix of accidental houses in the ShapeGraph for different parameters.

## `explainers`

Scripts to test implemented explainers.

## `metrics`
