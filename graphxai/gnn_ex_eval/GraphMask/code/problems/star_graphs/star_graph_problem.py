import numpy as np

from code.abstract.abstract_test_problem import AbstractTestProblem
from code.utils.evaluation.binary_classification_evaluator import BinaryClassificationEvaluator
import random
import torch
import pandas as pd
import scipy.sparse as sp
import torch
import os
import random
import pandas as pd
from scipy.spatial import distance_matrix


class StarGraphProblem(AbstractTestProblem):

    n_types = None
    n_leaves = None

    current_example = 0

    train_set_size = 10000
    dev_set_size = 500
    test_set_size = 500

    def __init__(self, configuration):
        self.n_types = configuration["task"]["n_colours"]
        self.n_leaves = [configuration["task"]["min_leaves"], configuration["task"]["max_leaves"]]

        self.evaluator = BinaryClassificationEvaluator()

        AbstractTestProblem.__init__(self, configuration)

    def count_raw_examples(self, split):
        if split == "train":
            return self.train_set_size
        elif split == "test":
            return self.test_set_size
        else:
            return self.dev_set_size

    def build_relationship(self, x, thresh=0.25):
        df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
        df_euclid = df_euclid.to_numpy()
        idx_map = []
        for ind in range(df_euclid.shape[0]):
            max_sim = np.sort(df_euclid[ind, :])[-2]
            neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
            import random
            random.seed(912)
            random.shuffle(neig_id)
            for neig in neig_id:
                if neig != ind:
                    idx_map.append([ind, neig])
        print('building edge relationship complete')
        idx_map = np.array(idx_map)

        return idx_map

    def load_bail(self, dataset, sens_attr="WHITE", predict_attr="RECID", path="../dataset/bail/", label_number=1000):

        print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)

        # build relationship
        if os.path.exists(f'{path}/{dataset}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
        else:
            edges_unordered = self.build_relationship(idx_features_labels[header], thresh=0.6)
            np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random
        random.seed(20)

        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
        for ind in idx_val:
            if ind in idx_train:
                print(ind)
            if ind in idx_test:
                print(ind)

        for ind in idx_test:
            if ind in idx_train:
                print(ind)
        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test, sens

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def build_example_org(self, id, split):
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "../dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = self.load_bail('bail', sens_attr,
                                                                              predict_attr, path=path_bail,
                                                                              label_number=label_number,
                                                                              )
        set_categorical_mask = []
        for i in range(features.shape[1]):
            if features[:, i].unique().shape[0] < 5:
                set_categorical_mask.append([i, 1, features[:, i].unique().shape[0]])
            else:
                set_categorical_mask.append([i, 0, 0])
        norm_features = self.feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        # adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, set_categorical_mask = model_data_loader.load_dataset()
        import ipdb
        ipdb.set_trace()
        return adj, features, labels, idx_train, idx_val

    def build_example(self, id, split):
        n_leaves = random.randint(*self.n_leaves)
        leaf_srcs = np.arange(1, n_leaves + 1)
        leaf_tgts = np.zeros_like(leaf_srcs)

        edges = np.stack((leaf_srcs, leaf_tgts)).transpose()

        edge_types = np.random.randint(self.n_types, size=len(edges))

        vertex_input = np.ones((edges[-1][0] + 1, 2 * self.n_types), dtype=np.int32) * -1
        xy = np.random.randint(self.n_types, size=2)
        while xy[0] == xy[1]:  # Naively repeat until we have two different numbers
            xy = np.random.randint(self.n_types, size=2)

        for k in range(1, len(vertex_input)):
            vertex_input[k] *= 0
            vertex_input[k][xy[0]] = 1
            vertex_input[k][xy[1] + self.n_types] = 1

        count_x = (edge_types == xy[0]).sum()
        count_y = (edge_types == xy[1]).sum()
        label = count_x > count_y

        attribution_labels = np.zeros_like(leaf_tgts)
        for i in range(len(attribution_labels)):
            if edge_types[i] == xy[0] or edge_types[i] == xy[1]:
                attribution_labels[i] = 1

        return vertex_input, edges, edge_types, label, attribution_labels

    def count_gnn_input_edges(self, batch):
        return sum([len(example[1]) for example in batch])

    def overwrite_labels(self, batch, predictions_to_overwrite_from):
        new_batch = []
        for i in range(len(batch)):
            example = batch[i]
            prediction = predictions_to_overwrite_from[i]
            actual_preds = prediction.get_predictions()[0]

            new_example = [x for x in example]
            new_example[3] = actual_preds

            new_batch.append(new_example)

        return new_batch

    def score_attributions(self, attribution, example):
        tp = 0
        fp = 0
        fn = 0

        attribution_labels = example[4]
        for i in range(len(attribution)):
            if attribution[i] == attribution_labels[i] and attribution_labels[i] == 1:
                tp += 1
            elif attribution[i] != attribution_labels[i] and attribution_labels[i] == 0:
                fp += 1
            elif attribution[i] != attribution_labels[i] and attribution_labels[i] == 1:
                fn += 1

        return tp, fp, fn