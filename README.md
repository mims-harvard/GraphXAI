# GraphXAI

First install the graphxai package from the root directory of this project:

  ```pip install -e .```
  
Then you will be able to run test scripts, access features within the package.

## How to Run Test Scripts
All test scripts are in the `test_scripts` folder in the main repo. Each script is associated with one or two explanation methods, and this is denoted by the naming of the files. For example, `test_GBP_MUTAG.py` and `test_GBP_BA.py` both run the Guided Backprop explainer while `test_CAM_MUTAG.py` and `test_CAM_BA.py` both run the Class-Activation Mapping (CAM) and Gradient-Class-Activation Mapping (Grad-CAM). The suffix `..._MUTAG.py` denotes that the script runs an explanation on the [MUTAG dataset](https://chrsmrrs.github.io/datasets/docs/datasets/) and `..._BA.py` denotes the script running an explanation on a custom version of the synthetic BA houses dataset, as is described in [GNNExplainer paper](https://arxiv.org/abs/1903.03894). 

To run the BA scripts, you'll simply run the script directly. For example, to run `test_CAM_BA.py`, the test script for CAM and Grad-CAM on the BA houses dataset, you'll run:

```>>> python3 test_CAM_BA.py```

To run the MUTAG scripts, you'll need to provide an additional command-line argument which corresponds to the index of the molecule for which you want to show an explanation. This is mainly for consistency across multiple runs of these scripts. For example, to run `test_GBP_MUTAG.py`, the test script for Guided Backprop on the MUTAG dataset, with the molecule at index 6, you'll run:

```>>> python3 test_GBP_MUTAG.py 6```

Note that all of these scripts will both provide output through stdout as well as show an image through matplotlib.

## Loading Saved ShapeGraph
ShapeGraph 1 is a large synthetic graph that has many desirable properties for evaluating GNN explanations. You load it in the following way:
```
from graphxai.datasets import load_ShapeGraph

SG = load_ShapeGraph(number=1) 
# SG now holds a fully loaded and verified ShapeGraph object.
```
