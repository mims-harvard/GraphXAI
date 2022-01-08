import torch
from torch_geometric.datasets import TUDataset

from graphxai.explainers import PGExplainer
from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import GCN, load_data, train, test


dataset = TUDataset(root='data/TUDataset', name='MUTAG')
mol_num = 2
train_loader, test_loader = load_data(dataset, mol_num)
model = GCN(in_channels=dataset.num_node_features, hidden_channels=64,
            out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    train(model, optimizer, criterion, train_loader)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

mol = dataset[mol_num]

atom_list = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
atom_map = {i:atom_list[i] for i in range(len(atom_list))}

atoms = []
for i in range(mol.x.shape[0]):
    atoms.append(atom_map[mol.x[i,:].tolist().index(1)])

batch = torch.zeros(1).type(torch.int64)
explainer = PGExplainer(model, explain_graph=True)
explainer.train_explanation_model(dataset, forward_kwargs={'batch': batch})
exp = explainer.get_explanation_graph(mol.x, mol.edge_index,
                                      forward_kwargs={'batch': batch})

model.eval()
pred = model(mol.x, mol.edge_index, batch=batch)

print('GROUND TRUTH LABEL: \t {}'.format(mol.y.item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=1).item()))

# exp = explainer.get_explanation_graph(mol.edge_index, mol.x, mol.y,
#                                       forward_args = (torch.zeros(1).type(torch.int64),))
