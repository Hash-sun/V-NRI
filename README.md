# Comparative study of a variant neural relational inference deep learning model and dynamical network analysis for p53-DNA allosteric interactions
**Abstract** <br>
Protein allostery is a critical regulatory mechanism in various biological processes, representing a challenging aspect of biological research. Dynamical network analysis serves as a foundational computational methodology for studying allosteric effects, while the recent emergence of neural relational inference (NRI) models has introduced novel insights into understanding it. In this study, we modified NRI model by integrating the multi-head self-attention module of transformer, and then compared the variant model with dynamical network analysis and initial NRI model for p53-DNA allosteric interactions. Our results show that the variant model is more focused on long-range allosteric than dynamical network in predicting p53-DNA allosteric pathways, and enhances superior accuracy and comprehensiveness compared to initial NRI model. Moreover, divergent allosteric pathways in wild-type (WT) versus mutant (MT) p53 may underlie their substantially different DNA recognition and binding behaviors. Finally, we found that even with undefined ending nodes, allosteric pathways consistently propagate from mutation sites toward DNA, and the length of pathways in MT p53 significantly is longer than that of WT p53. These suggest that mutation sites impair long-range allosteric communication, potentially disrupting signal transmission efficiency. Our results offer novel insights for a deep understanding of protein allosteric pathways through different methods.
![graphical abstract](https://github.com/user-attachments/assets/3d5ba1fa-a32b-4078-be44-299fe8a31515)

## Requirment
The environment can be configured using the file provided.


## data process
The training data are pdb files after molecular dynamics simulations. Place them in ```data/pdb``` folder.
```
python convert_dataset.py
```
It can seperate train, validation and test dataset.
You can download examples from *zenodo*. The website is placed in ```data/download```.

## Train
```
python main.py
```

## Test
```
python test.py
```
We provide two test samples in ```example``` folder. 


