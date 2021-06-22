# GraphMask

This repository contains an implementation of GraphMask, the interpretability technique for graph neural networks presented in our ICLR 2021 paper [Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking](https://arxiv.org/abs/2010.00577).

**Requirements**

We include a requirements.txt file for the specific environment we used to run the code. To run the code, please either set up your environment to match that, or verify that you have the following dependencies:

* Python 3
* PyTorch 1.8.1
* PyTorch Geometric 1.7
* AllenNLP 0.9.0
* SpaCy 2.1.9

**Running the Code**

We include models and interpreters for our synthetic task, for the question answering model by [De Cao et al. (2019)](https://www.aclweb.org/anthology/N19-1240/), and for the SRL model by [Marcheggiani and Titov (2017)](https://www.aclweb.org/anthology/D17-1159/).

To train a model, use our script by replacing \[configuration\] in the following with the appropriate file (default is the synthetic task, *configurations/star_graphs.json*):


```
python train_model.py --configuration \[configuration\]
```

Once you have trained the model, train and validate GraphMask by running:

```
python run_analysis.py --configuration \[configuration\]
```

For the synthetic task, you can optionally add a comparison between the performance of GraphMask and the faithfulness gold standard as follows:

```
python run_analysis.py --configuration \[configuration\] --gold_standard
```

To experiment with other analysis techniques, you can change the analysis strategy in the configuration file.


**Downloading Data**

For both the question answering and the SRL task, download the [840B Common Crawl GloVe embeddings](https://nlp.stanford.edu/projects/glove/) and place the file in *data/glove/*. For the question answering task, download the [Qangaroo dataset](http://qangaroo.cs.ucl.ac.uk/) and place the files in *data/qangaroo_v1.1/*. For the SRL task, follow the instructions [here](https://github.com/diegma/neural-dep-srl) to download the CoNLL-2009 dataset and generate vocabulary files. Place both dataset and vocabulary files in *data/conll2009/*.

**Citation**

Please cite our paper if you use this code in your own work:

```
@inproceedings{
   schlichtkrull2021interpreting,
   title={Interpreting Graph Neural Networks for {\{}NLP{\}} With Differentiable Edge Masking},
   author={Michael Sejr Schlichtkrull and Nicola De Cao and Ivan Titov},
   booktitle={International Conference on Learning Representations},
   year={2021},
   url={https://openreview.net/forum?id=WznmQa42ZAx}
}
```
