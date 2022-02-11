import glob
import torch


# Scans through explanations for RAND, adds random edge explanations

L = glob.glob('bigSG_explanations/RAND/*')

for f in L:
    exp = torch.load(f)

    # Add random edges:
    exp.edge_imp = torch.randn(exp.enc_subgraph.edge_index[0,:].shape)

    torch.save(exp, open(f, 'wb'))
