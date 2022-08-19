from torch_geometric.utils.subgraph import subgraph
from graphxai.datasets import ShapeGraph

SG = ShapeGraph(model_layers = 3, prob_connection = 0.1, num_subgraphs = 30, subgraph_size = 13)
data = SG.get_graph()

nidx, exp = SG.choose_node(inshape = True, label=1)

print('node index', nidx)

edge_mask = exp.edge_imp.bool()
nodes_in_mask = exp.enc_subgraph.nodes[exp.node_imp.long()]

print('masked nodes', nodes_in_mask)
print('part of edge index')

exp.visualize_node(num_hops = 3, graph_data = data, show = True)