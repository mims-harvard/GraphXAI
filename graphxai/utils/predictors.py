import torch


def correct_predictions(model, data, data_mask = None, get_mask = False):
    '''
    Get all indices of nodes that were correctly predicted by model
    '''

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    pred = out.softmax(dim=1).argmax(dim=1)  # Use the class with highest probability.

    if get_mask:
        if data_mask is None:
            correct = (pred == data.y)
        else:
            correct = (pred == data.y)[data_mask]
    else:
        if data_mask is None:
            correct = (pred == data.y).nonzero(as_tuple=True)[0]
        else:
            correct = ((pred == data.y) & data_mask).nonzero(as_tuple=True)[0]

    return correct