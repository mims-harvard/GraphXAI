## Towards a Rigorous Theoretical Analysis and Evaluation of GNN Explanations

This repository contains source code necessary to reproduce the main results of our NeurIPS'21 submission.

## 1. Setup

### Installing software
This repository is built using PyTorch. You can install the necessary libraries by pip installing the requirements text file `pip install -r ./requirements.txt`
After installing the packages from the requirements.txt, install the PyTorch Geometric packages following the instructions from [here.](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

**Note:** We ran our codes using python=3.7.9


## 2. Datasets
We ran our experiments on four real-world datasets from diverse domains (financial lending, criminal justice, citiation network). The datasets for _German credit graph_, _Recidivism graph_, and _Credit lending graph_ datasets are provided in the './datasets' folder. Due to space constraints the edge file of the credit dataset is zipped. The Ogbn-arxiv dataset is automatically downloaded while the codes are executed.

## 3. Usage
The main scripts running the experiments on the state-of-the-art GNN explanation methods are in [explain_model.py](explain_model.py), [graphmask_main.py](graphmask_main.py), [run_pgm_explainer.py](./PGMExplainer/proposed/run_pgm_explainer.py) and [run_pge_explainer.py](./DIG/run_pge_explainer.py).

### Examples
Script 1: Evaluate faithfulness, stability, and fairness of an explanation method for German Graph dataset using {Random Node Features, Random Edges, GNN Gradients, Integrated Gradients, GraphLIME, or GNNExplainer.

`python explain_model.py --model sage --seed 912 --var 1 --num_samples 50 --dataset german --algo grad`
<p align="left"><i>
  Faithfulness: 0.1848+-0.0095<br/>
  Stability: 0.2222+-0.0103<br/>
  Counterfactual Fairnss: 0.1371+-0.0072<br/>
  Fairness: 0.1538+-0.0124<br/>
</i></p>

Script 2: Evaluate faithfulness, stability, and fairness for GraphMASK explanation method on German Graph dataset.

`python graphmask_main.py --dataset german`
<p align="left"><i>
  Faithfulness: 0.0335+-0.0027<br/>
  Stability: 0.2701+-0.0076<br/>
  Counterfactual Fairnss: 0.0063+-0.0006<br/>
  Fairness: 0.0462+-0.0063<br/>
</i></p>  

Script 3: Evaluate faithfulness, stability, and fairness for PGMExplainer explanation method on German Graph dataset.

`python run_pgm_explainer.py --dataset german`
<p align="left"><i>
  Faithfulness: 0.1313+-0.0067<br/>
  Stability: 0.1828+-0.0060<br/>
  Counterfactual Fairnss: 0.1854+-0.0060<br/>
  Fairness: 0.1291+-0.0105<br/>
</i></p>   

Script 4: Evaluate faithfulness, stability, and fairness for PGExplainer explanation method on German Graph dataset.

`python run_pge_explainer.py`
<p align="left"><i>
  Faithfulness: 0.0764+-0.0064<br/>
  Stability: 0.3674+-0.0044<br/>
  Counterfactual Fairnss: 0.3577+-0.0096<br/>
  Fairness: 0.0780+-0.0083<br/>
</i></p>


Script 5: For similar results on Ogbn-arxiv dataset, use scripts [explain_arxiv_model.py](explain_arxiv_model.py), [arxiv_graphmask_main.py](arxiv_graphmask_main.py), [run_arxiv_pge_explainer.py](./DIG/run_arxiv_pge_explainer.py), and [arxiv_run_pgm_explainer.py](./PGMExplainer/proposed/arxiv_run_pgm_explainer.py).
