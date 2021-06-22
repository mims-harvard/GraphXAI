import networkx as nx
import math
import time
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from scipy import stats


class Node_Explainer:
    def __init__(
        self,
        model,
        A,
        X,
        ori_pred,
        num_layers,
        mode = 0,
        print_result = 1
    ):
        self.model = model
        self.model.eval()
        self.A = A
        self.X = X
        self.ori_pred = ori_pred
        self.num_layers = num_layers
        self.mode = mode
        self.print_result = print_result
        print("Explainer settings")
        print("\\ A dim: ", self.A.shape)
        print("\\ X dim: ", self.X.shape)
        print("\\ Number of layers: ", self.num_layers)
        print("\\ Perturbation mode: ", self.mode)
        print("\\ Print result: ", self.print_result)
    
    def n_hops_A(self, n_hops):
        # Compute the n-hops adjacency matrix
        adj = torch.tensor(self.A, dtype=torch.float)
        hop_adj = power_adj = adj
        for i in range(n_hops - 1):
            power_adj = power_adj @ adj
            prev_hop_adj = hop_adj
            hop_adj = hop_adj + power_adj
            hop_adj = (hop_adj > 0).float()
        return hop_adj.numpy().astype(int)
    
    def extract_n_hops_neighbors(self, nA, node_idx):
        # Return the n-hops neighbors of a node
        node_nA_row = nA[node_idx]
        neighbors = np.nonzero(node_nA_row)[0]
        node_idx_new = sum(node_nA_row[:node_idx])
        sub_A = self.A[neighbors][:, neighbors]
        sub_X = self.X[neighbors]
        return node_idx_new, sub_A, sub_X, neighbors
    
    def perturb_features_on_node(self,feature_matrix, node_idx, random = 0, mode = 0):
        # return a random perturbed feature matrix
        # random = 0 for nothing, 1 for random.
        # mode = 0 for random 0-1, 1 for scaling with original feature
        
        X_perturb = feature_matrix
        if mode == 0:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.random.randint(2, size = X_perturb[node_idx].shape[0])
            X_perturb[node_idx] = perturb_array
        elif mode == 1:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.multiply(X_perturb[node_idx],np.random.uniform(low=0.0, high=2.0, size = X_perturb[node_idx].shape[0]))
            X_perturb[node_idx] = perturb_array
        return X_perturb
    
    
    def explain(self, node_idx, num_samples = 100, top_node = None, p_threshold = 0.05, pred_threshold = 0.1):
        print("Explaining node: " + str(node_idx))
        nA = self.n_hops_A(self.num_layers)
        node_idx_new, sub_A, sub_X, neighbors = self.extract_n_hops_neighbors(nA,node_idx)
        
        if (node_idx not in neighbors):
            neighbors = np.append(neighbors, node_idx)
        
        X_torch = torch.tensor([self.X], dtype=torch.float)
        A_torch = torch.tensor([self.A], dtype=torch.float)
        pred_torch, _ = self.model.forward(X_torch, A_torch)
        soft_pred = np.asarray([softmax(np.asarray(pred_torch[0][node_].data)) for node_ in range(self.X.shape[0])])
        
        pred_node = np.asarray(pred_torch[0][node_idx].data)
        label_node = np.argmax(pred_node)
        soft_pred_node = softmax(pred_node)
        
        Samples = []
        Pred_Samples = []
        
        for iteration in range(num_samples):
            
            X_perturb = self.X.copy()
            sample = []
            for node in neighbors:
                seed = np.random.randint(2)
                if seed == 1:
                    latent = 1
                    X_perturb = self.perturb_features_on_node(X_perturb, node, random = seed)
                else:
                    latent = 0
                sample.append(latent)
            
            X_perturb_torch =  torch.tensor([X_perturb], dtype=torch.float)
            pred_perturb_torch, _ = self.model.forward(X_perturb_torch, A_torch)
            soft_pred_perturb = np.asarray([softmax(np.asarray(pred_perturb_torch[0][node_].data)) for node_ in range(self.X.shape[0])])

            sample_bool = []
            for node in neighbors:
                if (soft_pred_perturb[node,np.argmax(soft_pred[node])] + pred_threshold) < np.max(soft_pred[node]):
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)
            
            Samples.append(sample)
            Pred_Samples.append(sample_bool)
        
        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = Samples-Samples
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray([Samples[s,i]*10 + Pred_Samples[s,i]+1 for i in range(Samples.shape[1])])
            
        data = pd.DataFrame(Combine_Samples)
        ind_sub_to_ori = dict(zip(list(data.columns), neighbors))
        data = data.rename(columns={0: "A", 1: "B"}) # Trick to use chi_square test on first two data columns        
        ind_ori_to_sub = dict(zip(neighbors,list(data.columns)))
                
        p_values = []
        dependent_neighbors = []
        dependent_neighbors_p_values = []
        for node in neighbors:

            chi2, p = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [], data)
            p_values.append(p)
            if p < p_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)
        
        pgm_stats = dict(zip(neighbors,p_values))
  
        pgm_nodes = []
        if top_node == None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((top_node,len(neighbors)-1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [ind_sub_to_ori[node] for node in ind_top_p]
        
        data = data.rename(columns={"A": 0, "B": 1})
        data = data.rename(columns=ind_sub_to_ori)
        
        return pgm_nodes, data, pgm_stats


    def explain_range(self, node_list, num_samples = 1000, top_node = None, p_threshold = 0.05, pred_threshold = 0.1):
        nA = self.n_hops_A(self.num_layers)
        
        neighbors_list = {}
        all_neighbors = []
        for node in node_list:
            _,_,_,neighbors = self.extract_n_hops_neighbors(nA,node)
            if (node not in neighbors):
                neighbors = np.append(neighbors, node)
            neighbors_list[node] = neighbors
            all_neighbors = list(set(all_neighbors)| set(np.append(neighbors, node)))
            
        X_torch = torch.tensor([self.X], dtype=torch.float)
        A_torch = torch.tensor([self.A], dtype=torch.float)
        pred_torch, _ = self.model.forward(X_torch, A_torch)
        soft_pred = np.asarray([softmax(np.asarray(pred_torch[0][node_].data)) for node_ in range(self.X.shape[0])])
        
        Samples = []
        Pred_Samples = []
        
        for iteration in range(num_samples):
            
            X_perturb = self.X.copy()
            sample = []
            for node in all_neighbors:
                seed = np.random.randint(2)
                if seed == 1:
                    latent = 1
                    X_perturb = self.perturb_features_on_node(X_perturb, node, random = seed, mode = self.mode)
                else:
                    latent = 0
                sample.append(latent)
            
            X_perturb_torch =  torch.tensor([X_perturb], dtype=torch.float)
            pred_perturb_torch, _ = self.model.forward(X_perturb_torch, A_torch)
            soft_pred_perturb = np.asarray([softmax(np.asarray(pred_perturb_torch[0][node_].data)) for node_ in range(self.X.shape[0])])

            sample_bool = []
            for node in all_neighbors:
                if (soft_pred_perturb[node,np.argmax(soft_pred[node])] + pred_threshold) < np.max(soft_pred[node]):
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)
            
            Samples.append(sample)
            Pred_Samples.append(sample_bool)
        
        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = Samples-Samples
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray([Samples[s,i]*10 + Pred_Samples[s,i]+1 for i in range(Samples.shape[1])])
            
        data = pd.DataFrame(Combine_Samples)
        data = data.rename(columns={0: "A", 1: "B"}) # Trick to use chi_square test on first two data columns
        ind_sub_to_ori = dict(zip(list(data.columns), all_neighbors))
        ind_ori_to_sub = dict(zip(all_neighbors,list(data.columns)))
                
        explanations = {}
        for target in node_list:
            print("Generating explanation for node: ", target)
            
            p_values = []
            dependent_neighbors = []
            dependent_neighbors_p_values = []
            for node in neighbors_list[target]:
                p = 0
                if node == target:
                    p = 0
                    p_values.append(p)
                else:
                    chi2, p = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[target], [], data)
                    p_values.append(p)
                if p < 0.05:
                    dependent_neighbors.append(node)
                    dependent_neighbors_p_values.append(p)
                    
            pgm_nodes = []
            if top_node == None:
                pgm_nodes = dependent_neighbors
            else:
                ind_subnei_to_ori = dict(zip(range(len(neighbors_list[target])), neighbors_list[target]))
                if top_node < len(neighbors_list[target]):
                    ind_top = np.argpartition(p_values, top_node)[0:top_node]
                    pgm_nodes = [ind_subnei_to_ori[node] for node in ind_top]
                else:
                    pgm_nodes = neighbors_list[target]
           
            explanations[target] = pgm_nodes
            if self.print_result == 1:
                print(pgm_nodes)

        return explanations
    
    def generalize_target(self, x):
        if x > 10:
            return x - 10
        else:
            return x

    def generalize_others(self, x):
        if x == 2:
            return 1
        elif x == 12:
            return 11
        else:
            return x

    def generate_evidence(self, evidence_list):
        return dict(zip(evidence_list,[1 for node in evidence_list]))
    
    def chi_square(self, X, Y, Z, data):
        """
        Modification of Chi-square conditional independence test from pgmpy
        Tests the null hypothesis that X is independent from Y given Zs.

        Parameters
        ----------
        X: int, string, hashable object
            A variable name contained in the data set
        Y: int, string, hashable object
            A variable name contained in the data set, different from X
        Zs: list of variable names
            A list of variable names contained in the data set, different from X and Y.
            This is the separating set that (potentially) makes X and Y independent.
            Default: []
        Returns
        -------
        chi2: float
            The chi2 test statistic.
        p_value: float
            The p_value, i.e. the probability of observing the computed chi2
            statistic (or an even higher value), given the null hypothesis
            that X _|_ Y | Zs.
        sufficient_data: bool
            A flag that indicates if the sample size is considered sufficient.
            As in [4], require at least 5 samples per parameter (on average).
            That is, the size of the data set must be greater than
            `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
            (c() denotes the variable cardinality).
        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.2.2.3 (page 789)
        [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
        [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4
        """
        X = str(int(X))
        Y = str(int(Y))
        if isinstance(Z, (frozenset, list, set, tuple)):
            Z = list(Z)        
        Z = [str(int(z)) for z in Z]
            
        state_names = {
            var_name: data.loc[:, var_name].unique() for var_name in data.columns
        }
        
        row_index = state_names[X]
        column_index = pd.MultiIndex.from_product(
                [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
            )
        
        XYZ_state_counts = pd.crosstab(
                    index=data[X], columns= [data[Y]] + [data[z] for z in Z],
                    rownames=[X], colnames=[Y] + Z
                )

        if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
                XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
        XYZ_state_counts = XYZ_state_counts.reindex(
                index=row_index, columns=column_index
            ).fillna(0)
        
        if Z:
            XZ_state_counts = XYZ_state_counts.sum(axis=1,level = list( range(1,len(Z)+1)) )  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Z)  # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)
            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both
        
        XYZ_expected = np.zeros(XYZ_state_counts.shape)

        r_index = 0
        for X_val in XYZ_state_counts.index:
            X_val_array = []
            if Z:
                for Y_val in XYZ_state_counts.columns.levels[0]:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + list(temp.to_numpy())
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index=+1
            else:
                for Y_val in XYZ_state_counts.columns:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + [temp]
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index=+1
        
        observed = XYZ_state_counts.to_numpy().reshape(1,-1)
        expected = XYZ_expected.reshape(1,-1)
        observed, expected = zip(*((o, e) for o, e in zip(observed[0], expected[0]) if not (e == 0 or math.isnan(e) )))
        chi2, significance_level = stats.chisquare(observed, expected)

        return chi2, significance_level
    
    def search_MK(self, data, target, nodes):
        target = str(int(target))
        data.columns = data.columns.astype(str)
        nodes = [str(int(node)) for node in nodes]
        
        MB = nodes
        while True:
            count = 0
            for node in nodes:
                evidences = MB.copy()
                evidences.remove(node)    
                _, p = self.chi_square(target, node, evidences, data[nodes+ [target]])
                if p > 0.05:
                    MB.remove(node)
                    count = 0
                else:
                    count = count + 1
                    if count == len(MB):
                        return MB
    
    def pgm_generate(self, target, data, pgm_stats, subnodes, child = None):
   
        subnodes = [str(int(node)) for node in subnodes]
        target = str(int(target))
        subnodes_no_target = [node for node in subnodes if node != target]
        data.columns = data.columns.astype(str)
        
        MK_blanket = self.search_MK(data, target, subnodes_no_target.copy())
        

        if child == None:
            est = HillClimbSearch(data[subnodes_no_target], scoring_method=BicScore(data))
            pgm_no_target = est.estimate()
            for node in MK_blanket:
                if node != target:
                    pgm_no_target.add_edge(node,target)

        #   Create the pgm    
            pgm_explanation = BayesianModel()
            for node in pgm_no_target.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_no_target.edges():
                pgm_explanation.add_edge(edge[0],edge[1])

        #   Fit the pgm
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)
        else:
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
                
            est = HillClimbSearch(data_ex, scoring_method=BicScore(data_ex))
            pgm_w_target_explanation = est.estimate()
            
            #   Create the pgm    
            pgm_explanation = BayesianModel()
            for node in pgm_w_target_explanation.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm_explanation.add_edge(edge[0],edge[1])

            #   Fit the pgm
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)
        

        return pgm_explanation
    
    def pgm_conditional_prob(self, target, pgm_explanation, evidence_list):
        pgm_infer = VariableElimination(pgm_explanation)
        for node in evidence_list:
            if node not in list(pgm_infer.variables):
                print("Not valid evidence list.")
                return None
        evidences = self.generate_evidence(evidence_list)
        elimination_order = [node for node in list(pgm_infer.variables) if node not in evidence_list]
        elimination_order = [node for node in elimination_order if node != target]
        q = pgm_infer.query([target], evidence = evidences,
                        elimination_order = elimination_order, show_progress=False)
        return q.values[0]
        
