import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.nn import MessagePassing
from typing import Tuple


def extract_step(model: nn.Module, x: Tensor, edge_index: Tensor, detach: bool = True, forward_kwargs: dict = {}):
    '''Gets information about every layer in the graph
    Args:

        forward_kwargs (tuple, optional): Additional arguments to model forward call (other than x and edge_index)
            (default: :obj:`None`)
    '''

    layer_extractor = []
    hooks = []

    def register_hook(module: nn.Module):
        if not list(module.children()) or isinstance(module, MessagePassing):
            hooks.append(module.register_forward_hook(forward_hook))

    def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
        # input contains x and edge_index
        if detach:
            layer_extractor.append((module, input[0].clone().detach(), output.clone().detach()))
        else:
            layer_extractor.append((module, input[0], output))

    # --- register hooks ---
    model.apply(register_hook)

    # ADDED: OWEN QUEEN --------------
    if forward_kwargs is None:
        _ = model(x, edge_index)
    else:
        _ = model(x, edge_index, **forward_kwargs)
    # --------------------------------
    # Remove hooks:
    for hook in hooks:
        hook.remove()

    # --- divide layer sets ---

    walk_steps = []
    fc_steps = []
    step = {'input': None, 'module': [], 'output': None}
    for layer in layer_extractor:
        if isinstance(layer[0], MessagePassing):
            if step['module']: # Append step that had previously been building
                walk_steps.append(step)

            step = {'input': layer[1], 'module': [], 'output': None}

        elif isinstance(layer[0], GNNPool):
            pool_flag = True
            if step['module']:
                walk_steps.append(step)

            # Putting in GNNPool
            step = {'input': layer[1], 'module': [], 'output': None}

        elif isinstance(layer[0], nn.Linear):
            if step['module']:
                if isinstance(step['module'][0], MessagePassing):
                    walk_steps.append(step) # Append MessagePassing layer to walk_steps
                else: # Always append Linear layers to fc_steps
                    fc_steps.append(step)

            step = {'input': layer[1], 'module': [], 'output': None}

        # Also appends non-trainable layers to step (not modifying input):
        step['module'].append(layer[0])
        step['output'] = layer[2]

    if step['module']:
        if isinstance(step['module'][0], MessagePassing):
            walk_steps.append(step)
        else: # Append anything to FC that is not MessagePassing at its origin
            # Still supports sequential layers
            fc_steps.append(step)

    for walk_step in walk_steps:
        if hasattr(walk_step['module'][0], 'nn') and walk_step['module'][0].nn is not None:
            # We don't allow any outside nn during message flow process in GINs
            walk_step['module'] = [walk_step['module'][0]]

    return walk_steps, fc_steps


def get_activation(model: torch.nn.Module, x: torch.Tensor,
                   edge_index: torch.Tensor, forward_kwargs: dict = {}):
    """
    Get the activation of each layer of the GNN model.
    """
    activation = {}
    def get_activation():
        def hook(model, inp, out):
            activation['layer'] = out.detach()
        return hook

    layer.register_forward_hook(get_activation())

    with torch.no_grad():
        _ = model(x, edge_index, **forward_kwargs)

    return activation['layer']
