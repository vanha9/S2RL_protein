# S2RL_protein
Structure-Guided Sequence Representation Learning for Generalizable Protein Function Prediction  

![S2RL main pipeline](Structure_Figure16.png)

## Dataset
The dataset can be found at [HEAL GitHub Repository](https://github.com/ZhonghuiGu/HEAL).

## Preprocessing
To add Laplacian eigenvectors to each split graph dataset, run the Jupyter notebook:  `laplacian_pre.ipynb`

## Experiment
You can customize the training behavior by modifying variables in train_script0.sh. The script accepts the following variables in order:

**num_node**: Number of super-nodes used in the graph model.  
**device_num**: CUDA device number for training.  
**evec_ratio**: The ratio of Laplacian eigenvectors to use, as described in the paper.  
**num_head**: Number of heads used in the multi-head attention module.  

These variables allow you to flexibly control the model architecture and training settings.

**Training**

You can start the training process using:  `train_script0.sh`

**Evaluation**

After training, you can perform testing and evaluation with:  `test_script0.sh`
