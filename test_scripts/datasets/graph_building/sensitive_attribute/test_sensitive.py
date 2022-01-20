from graphxai.datasets import ShapeGraph

SG = ShapeGraph(
    model_layers = 3,
    num_subgraphs = 100,
    prob_connection = 0.08,
    subgraph_size = 10,
    class_sep = 5,
    n_informative = 6,
    n_clusters_per_class = 1,
    verify = False,
    make_explanations=False
)

data = SG.get_graph()

print('All X:\n', data.x[34:56])

print('My sensitive location', SG.sensitive_feature)

print('Sensitive', data.x[34:56,SG.sensitive_feature])