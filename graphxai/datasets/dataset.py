import pickle, random
import torch
from copy import deepcopy
from typing import Tuple

import numpy as np
from torch_geometric.utils.convert import to_networkx
from graphxai.utils import Explanation

from torch_geometric.data import Dataset, data
from torch_geometric.loader import DataLoader
#from torch.utils.data.dataloader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import train_test_split

from typing import List, Optional, Callable, Union, Any, Tuple

from graphxai.utils.explanation import EnclosingSubgraph
from graphxai.utils import to_networkx_conv
import graphxai.datasets as gxai_data

def get_dataset(dataset, download = False):
    '''
    Base function for loading all datasets.
    Args:
        dataset (str): Name of dataset to retrieve.
        download (bool): If `True`, downloads dataset to local directory.
    '''

    if dataset == 'MUTAG':
        # Get MUTAG dataset
        return gxai_data.real_world.MUTAG.MUTAG()
    else:
        raise NameError("Cannot find dataset '{}'.".format(dataset))

class NodeDataset:
    def __init__(self, 
        name, 
        num_hops: int,
        download: Optional[bool] = False,
        root: Optional[str] = None
        ):
        self.name = name
        self.num_hops = num_hops

        #self.graph = WholeGraph() # Set up whole_graph with all None
        #self.explanations = []

    def get_graph(self, 
        use_fixed_split: bool = True, 
        split_sizes: Tuple = (0.7, 0.2, 0.1),
        stratify: bool = True, 
        seed: int = None):
        '''
        Gets graph object for training/validation/testing purposes
            - Sets masks within torch_geometric.data.Data object
        Args:
            use_static_split (bool, optional): If true, uses the fixed train/val/test
                mask defined by the child dataset class. (:default: :obj:`True`)
            split_sizes (tuple, length 2 or 3, optional): If length 2, index 0 is train 
                size, index 1 is test size. If length 3, index 2 becomes val size. Does 
                not need to sum to 1, just needs to capture relative proportions of each 
                split. (:default: :obj:`(0.7, 0.2, 0.1)`)
            stratify (bool, optional): If True, stratifies the splits by class label. 
                Only relevant if `use_fixed_split = False`. (:default: :obj:`True`)
            seed (int, optional): Seed for splitting. (:default: :obj:`None`)            
    
        :rtype: torch_geometric.data.Data
        Returns:
            graph: Data object containing masks over the splits (graph.train_mask, 
                graph.valid_mask, graph.test_mask) and the full data for the graph.
        '''

        if sum(split_sizes) != 1: # Normalize split sizes
            split_sizes = np.array(split_sizes) / sum(split_sizes)

        if use_fixed_split:
            # Set train, test, val static masks:
            self.graph.train_mask = self.fixed_train_mask
            self.graph.valid_mask = self.fixed_valid_mask
            self.graph.test_mask  = self.fixed_test_mask

        else:
            #assert sum(split_sizes) == 1, "split_sizes must sum to 1"
            assert len(split_sizes) == 3, "split_sizes must contain (train_size, test_size, valid_size)"
            # Create a split for user (based on seed, etc.)
            train_mask, test_mask = train_test_split(list(range(self.graph.num_nodes)), 
                                test_size = split_sizes[1] + split_sizes[2], 
                                random_state = seed, stratify = self.graph.y.tolist() if stratify else None)
            # print(self.graph.y.tolist())
            # print(train_mask)
            # exit(0)

            if split_sizes[2] > 0:
                valid_mask, test_mask = train_test_split(test_mask, 
                                    test_size = split_sizes[2] / split_sizes[1],
                                    random_state = seed, stratify = self.graph.y[test_mask].tolist() if stratify else None)
                self.graph.valid_mask = torch.tensor([i in valid_mask for i in range(self.graph.num_nodes)], dtype = torch.bool)

            self.graph.train_mask = torch.tensor([i in train_mask for i in range(self.graph.num_nodes)], dtype = torch.bool)
            self.graph.test_mask  = torch.tensor([i in test_mask  for i in range(self.graph.num_nodes)], dtype = torch.bool)

        return self.graph

    def download(self):
        '''TODO: Implement'''
        pass

    def get_enclosing_subgraph(self, node_idx: int):
        '''
        Args:
            node_idx (int): Node index for which to get subgraph around
        '''
        k_hop_tuple = k_hop_subgraph(node_idx, 
            num_hops = self.num_hops, 
            edge_index = self.graph.edge_index)
        return EnclosingSubgraph(*k_hop_tuple)

    def nodes_with_label(self, label = 0, mask = None) -> torch.Tensor:
        '''
        Get all nodes that are a certain label
        Args:
            label (int, optional): Label for which to find nodes.
                (:default: :obj:`0`)

        Returns:
            torch.Tensor: Indices of nodes that are of the label
        '''
        if mask is not None:
            return ((self.graph.y == label) & (mask)).nonzero(as_tuple=True)[0]
        return (self.graph.y == label).nonzero(as_tuple=True)[0]

    def choose_node_with_label(self, label = 0, mask = None) -> Tuple[int, Explanation]:
        '''
        Choose a random node with a given label
        Args:
            label (int, optional): Label for which to find node.
                (:default: :obj:`0`)

        Returns:
            tuple(int, Explanation):
                int: Node index found
                Explanation: explanation corresponding to that node index
        '''
        nodes = self.nodes_with_label(label = label, mask = mask)
        node_idx = random.choice(nodes).item()

        return node_idx, self.explanations[node_idx]

    def nodes_in_shape(self, inshape = True, mask = None):
        '''
        Get a group of nodes by shape membership.

        Args:
            inshape (bool, optional): If the nodes are in a shape.
                :obj:`True` means that the nodes returned are in a shape.
                :obj:`False` means that the nodes are not in a shape.

        Returns:
            torch.Tensor: All node indices for nodes in or not in a shape.
        '''
        # Get all nodes in a shape
        condition = (lambda n: self.G.nodes[n]['shape'] > 0) if inshape \
                else (lambda n: self.G.nodes[n]['shape'] == 0)

        if mask is not None:
            condition = (lambda n: (condition(n) and mask[n].item()))

        return torch.tensor([n for n in self.G.nodes if condition(n)]).long()

    def choose_node_in_shape(self, inshape = True, mask = None) -> Tuple[int, Explanation]:
        '''
        Gets a random node by shape membership.

        Args:
            inshape (bool, optional): If the node is in a shape.
                :obj:`True` means that the node returned is in a shape.
                :obj:`False` means that the node is not in a shape.

        Returns:
            Tuple[int, Explanation]
                int: Node index found
                Explanation: Explanation corresponding to that node index
        '''
        nodes = self.nodes_in_shape(inshape = inshape, mask = mask)
        node_idx = random.choice(nodes).item()

        return node_idx, self.explanations[node_idx]


    def choose_node(self, inshape = None, label = None, split = None):
        '''
        Chooses random nodes in the graph. Has support for multiple logical
            indexing.

        Args:
            inshape (bool, optional): If the node is in a shape.
                :obj:`True` means that the node returned is in a shape.
                :obj:`False` means that the node is not in a shape.
            label (int, optional): Label for which to find node.
                (:default: :obj:`0`)
        
        Returns:
        '''
        split = split.lower() if split is not None else None

        if split == 'validation' or split == 'valid' or split == 'val':
            split = 'val'

        map_to_mask = {
            'train': self.graph.train_mask,
            'val': self.graph.valid_mask,
            'test': self.graph.test_mask,
        }
        
        # Get mask based on provided string:
        mask = None if split is None else map_to_mask[split]

        if inshape is None:
            if label is None:
                to_choose = torch.arange(end = self.num_nodes)
            else:
                to_choose = self.nodes_with_label(label = label, mask = mask)
        
        elif label is None:
            to_choose = self.nodes_in_shape(inshape = inshape, mask = mask)

        else:
            t_inshape = self.nodes_in_shape(inshape = inshape, mask = mask)
            t_label = self.nodes_with_label(label = label, make = mask)

            # Joint masking over shapes and labels:
            to_choose = torch.as_tensor([n.item() for n in t_label if n in t_inshape]).long()

        assert_fmt = 'Could not find a node in {} with inshape={}, label={}'
        assert to_choose.nelement() > 0, assert_fmt.format(self.name, inshape, label)

        node_idx = random.choice(to_choose).item()
        return node_idx, self.explanations[node_idx]

    def __len__(self) -> int:
        return 1 # There is always just one graph

    def dump(self, fname = None):
        fname = self.name + '.pickle' if fname is None else fname
        torch.save(self, open(fname, 'wb'))
    @property
    def x(self):
        return self.graph.x

    @property
    def edge_index(self):
        return self.graph.edge_index

    @property
    def y(self):
        return self.graph.y

    def __getitem__(self, idx):
        assert idx == 0, 'Dataset has only one graph'
        return self.graph, self.explanation


class GraphDataset:
    def __init__(self, name, split_sizes = (0.7, 0.2, 0.1), seed = None, device = None):

        self.name = name

        self.seed = seed
        self.device = device
        # explanation_list - list of explanations for each graph

        if split_sizes[1] > 0:
            self.train_index, self.test_index = train_test_split(torch.arange(start = 0, end = len(self.graphs)), 
                test_size = split_sizes[1] + split_sizes[2], random_state=self.seed, shuffle = False)
        else:
            self.test_index = None
            self.train_index = torch.arange(start = 0, end = len(self.graphs))

        if split_sizes[2] > 0:
            self.test_index, self.val_index = train_test_split(self.test_index, 
                test_size = split_sizes[2] / (split_sizes[1] + split_sizes[2]),
                random_state = self.seed, shuffle = False)

        else:
            self.val_index = None

        self.Y = torch.tensor([self.graphs[i].y for i in range(len(self.graphs))]).to(self.device)

    def get_data_list(
            self,
            index,
        ):
        data_list = [self.graphs[i].to(self.device) for i in index]
        exp_list = [self.explanations[i] for i in index]

        return data_list, exp_list

    def get_loader(
            self, 
            index,
            batch_size = 16,
            **kwargs
        ):

        data_list, exp_list = self.get_data_list(index)

        for i in range(len(data_list)):
            data_list[i].exp_key = [i]

        loader = DataLoader(data_list, batch_size = batch_size, shuffle = True)

        return loader, exp_list

    def get_train_loader(self, batch_size = 16):
        return self.get_loader(index=self.train_index, batch_size = batch_size)

    def get_train_list(self):
        return self.get_data_list(index = self.train_index)

    def get_test_loader(self):
        assert self.test_index is not None, 'test_index is None'
        return self.get_loader(index=self.test_index, batch_size = 1)

    def get_test_list(self):
        assert self.test_index is not None, 'test_index is None'
        return self.get_data_list(index = self.test_index)

    def get_val_loader(self):
        assert self.test_index is not None, 'val_index is None'
        return self.get_loader(index=self.val_index, batch_size = 1)

    def get_val_list(self):
        assert self.val_index is not None, 'val_index is None'
        return self.get_data_list(index = self.val_index)

    def get_train_w_label(self, label):
        inds_to_choose = (self.Y[self.train_index] == label).nonzero(as_tuple=True)[0]
        in_train_idx = inds_to_choose[torch.randint(low = 0, high = inds_to_choose.shape[0], size = (1,))]
        chosen = self.train_index[in_train_idx.item()]

        return self.graphs[chosen], self.explanations[chosen]

    def get_test_w_label(self, label):
        assert self.test_index is not None, 'test_index is None'
        inds_to_choose = (self.Y[self.test_index] == label).nonzero(as_tuple=True)[0]
        in_test_idx = inds_to_choose[torch.randint(low = 0, high = inds_to_choose.shape[0], size = (1,))]
        chosen = self.test_index[in_test_idx.item()]

        return self.graphs[chosen], self.explanations[chosen]

    def get_graph_as_networkx(self, graph_idx):
        '''
        Get a given graph as networkx graph
        '''

        g = self.graphs[graph_idx]
        return to_networkx_conv(g, node_attrs = ['x'], to_undirected=True)

    def download(self):
        pass

    def __getitem__(self, idx):
        return self.graphs[idx], self.explanations[idx]

    def __len__(self):
        return len(self.graphs)
