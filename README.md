# Comparative study of a variant neural relational inference deep learning model and dynamical network analysis for p53-DNA allosteric interactions
**Abstract** <br>
Protein allostery is a critical regulatory mechanism in various biological processes, representing a challenging aspect of biological research. Dynamical network analysis serves as a foundational computational methodology for studying allosteric effects, while the recent emergence of neural relational inference (NRI) models has introduced novel insights into understanding it. In this study, we modified NRI model by integrating the multi-head self-attention module of transformer, and then compared the variant model with dynamical network analysis and initial NRI model for p53-DNA allosteric interactions. Our results show that the variant model is more focused on long-range allosteric than dynamical network in predicting p53-DNA allosteric pathways, and enhances superior accuracy and comprehensiveness compared to initial NRI model. Moreover, divergent allosteric pathways in wild-type (WT) versus mutant (MT) p53 may underlie their substantially different DNA recognition and binding behaviors. Finally, we found that even with undefined ending nodes, allosteric pathways consistently propagate from mutation sites toward DNA, and the length of pathways in MT p53 significantly is longer than that of WT p53. These suggest that mutation sites impair long-range allosteric communication, potentially disrupting signal transmission efficiency. Our results offer novel insights for a deep understanding of protein allosteric pathways through different methods.
![graphical abstract](https://github.com/user-attachments/assets/3d5ba1fa-a32b-4078-be44-299fe8a31515)

## Requirment
The repo mainly requires the following packages.
* Python >= 3.7
* Pytorch >= 1.2
* numpy == 1.26
* pandas == 2.2.3
* networkx 2.5

In addition, the environment can be configured using the file provided.
You can install the environment through the YML file with the following command：
```
conda env create -f ppi.yaml
```
```
conda activate ppi
```



## data process
The training data are pdb files after molecular dynamics simulations. Place them in ```data/pdb``` folder.
```
python convert_dataset.py
```
It can seperate train, validation and test dataset.
You can download examples from *zenodo*. The website is https://doi.org/10.5281/zenodo.15687924.

## Train
You can train the model with the following command
```
python main.py
```
These are the main arguments：
* ```--num-residues``` - Number of residues of the PDB. 
* ```--save-folder``` - Where to save the trained model, leave empty to not save anything. default='logs'
* ```--epochs``` - Number of epochs to train.default=100
* ```--encoder``` - Type of path encoder model (mlp , cnn or mlpatten).
* ```--decoder``` - Type of decoder model (mlp, rnn, or sim).
* ```--number-exp``` - number of experiments.


If you want to modify epoch and encoder, you can do so by:
```
python main.py --epoch 500 --encoder mlp
```
## Test
```
python test.py
```
The main arguments are similar to the main.py, and you can change the them.
We provide two test samples in ```example``` folder. 
You can download the version of model that we trained from https://doi.org/10.5281/zenodo.15687924


