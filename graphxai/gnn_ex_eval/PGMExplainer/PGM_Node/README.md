# PGM-Explainer for Node classification

This folder contains the source code of using **PGM-Explainer** to explain node classification.

Example explaination:

<p align="center">
  <img src="https://github.com/vunhatminh/PGMExplainer/blob/master/PGM_Node/Explain_GNN/view/pgm_304.jpg"/>
</p>

Folder description:
  * Generate_XA_Data: To generate the graph data for the node classification experiments
  * Train_GNN_model: To train the GNN models for the  node classification tasks
  * Explain_GNN: To generate PGM Explaination for GNN predictions
  
To generate graph data, direct into Generate_XA_Data and run: 

`python3 GenData.py --dataset [dataset-name]` 

  * dataset-name: syn1, syn2, syn3, syn4, syn5, syn6, bitcoinalpha, bitcoinotc       
  * Generate feature matrix X, adjacency matrix A and ground-truth label L are stored in "XAL" folder
  * The synthetic data 1,2,3,4 and 5 are from https://github.com/RexYing/gnn-model-explainer

To generate ground-truth for explanations, direct into Generate_XA_Data and run: 

`python3 GenGroundTruth.py --dataset [dataset-name]`
  
  * dataset-name: bitcoinalpha, bitcoinotc       
  * Generated ground-truth Explanations are saved in "ground_truth_explanation" folder

To train GNN model, direct into Train_GNN_model and run:

`python3 train.py --dataset [dataset-name]`

  * dataset-name: syn1, syn2, syn3, syn4, syn5, syn6, bitcoinalpha, bitcoinotc
  * Generated model states are saved in "cpkt" folders
  * The model are obtained from https://github.com/RexYing/gnn-model-explainer

To run PGM explainer, direct into Explain_GNN folder and run:

`python3 main.py --dataset [dataset-name] --num-perturb-samples [int1] --top-node [int2]`
   
   * dataset-name: syn1, syn2, syn3, syn4, syn5, syn6, bitcoinalpha, bitcoinotc
   * int1: recommend 800-1000
   * int2: recommend None or 3,4,5
   * explanations are stored in result folder. Might need to creat 'dataset-name' folder to properly stored the result.

To evaluate explanations, direct into Explain_GNN folder and run:

`python3 evaluate_explanations.py --dataset [dataset-name] --num-perturb-samples [int1] --top-node [int2]`
 
   * dataset-name: syn1, syn2, syn3, syn4, syn5, syn6, bitcoinalpha, bitcoinotc
   * int1: int1 used in generating PGM explanations
   * int2: int2 used in generating PGM explanations
   * reports are stored in result folder

We also include the notebook *notebook_example_on_syn_6.ipynb* as an example.

Reported Results:

<p align="center">
  <img src="https://github.com/vunhatminh/PGMExplainer/blob/master/PGM_Node/Explain_GNN/result/syn_result.png"/>
</p>

<p align="center">
  <img src="https://github.com/vunhatminh/PGMExplainer/blob/master/PGM_Node/Explain_GNN/result/bitcoin_result.png"/>
</p>
