# PPT-GNN: A Practical Pretrained Spatio-Temporal Graph Neural Network for Network Security

# Table of contents
1. [Project Description](#project-description)
2. [How To Run](#how_to_run)

## Project Description
A Graph Neural Network (GNN)-based Network Intrusion Detection (NIDS) Model that operates under **more practical, realistic conditions**: 
- NF-based datasets.
- Small time-windows for computationaly efficient, near-real-time inference.
- Highly expressive spatio-temporal framework.
- Unsupervised pretraining for reduced labeled data dependency.

Code repository of paper currently under review but available in ArXiv: https://arxiv.org/pdf/2406.13365

## How To Run
1. Extract (in-place): **data/raw/raw_datasets.zip**
2. Run only cell in notebook: **ingest_data.ipynb**
3. For pretraining open notebook: **pretraining.ipynb**
   
    3.1. Go to cell in section *set hyperparameters* to set parameters of choice.
   
    3.2. Run cell in section *pretraining* to run pretraining routine
   
4. To replicate each experiment training, open notebook: **train_nids.ipynb**
   
    4.1. Train baseline GNNs from scratch by running cells under *Experiment 1: General GNNs from scratch Parameter Optimization*
  
    4.2. Fine tune a pretrained GNN by running cells under *Experiment 2: GNNs from pre-trained*
  
    4.3. Train MLP baseline by running cells under *Experiment 3: Reference MLP Baselines*
  
5. To evaluate all models/experiment on test set, open notebook: **train_nids.ipynb**
   
    5.1. Run all cells under *Evaluation* 


