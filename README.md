# AMI-Net+

This is the source code of our ICONIP 2019 submitted paper: "[AMI-Net+: A Novel Multi-Instance Neural Network for Medical 
Diagnosis from Incomplete and Imbalanced Data](https://arxiv.org/abs/1907.01734)". Please cite it if you use AMI-Net+ for 
your research.


## Introduction

* AMI-Net+ is the improvement of our previous work, AMI-Net, to further resolve the incomplete and imbalanced data. We change the cross-entropy to the focal loss and propose a self-adaptive pooling on the instance-level. The overall architecture shows here. Enjoy.

<br/>
<div align="middle"><img src="https://github.com/Zeyuan-Wang/AMI-Netv2/blob/master/img/AMI-Net+.png"width="70%"></div>
 
 
## Data Description

* **sample_data.xlsx**: There are 3000 sample cases randomly extracted from the whole "Western Medicine" dataset. The column "y" is the binary prediction target. The other 88 columns are all binary features, i.e., whether the patient has this symptom. And for each case, there exist at most 21 features and 5 at least, representing the individual patient condition.


## File Description

* **main.py**:  The main code for running the model.
* **config.py**:  Contains the hyper-parameters of AMI-Net+.
* **load_data.py**:  The script for data transformation.
* **model.py**:  The script for AMI-Net+ construction.
* **utils.py**:  Contains the computational modules of the AMI-Net+.


## Dependencies

* Python  3.6.8
* numpy  1.16.0
* sklearn  0.18.1
* tensorflow  2.0.0-beta1
```shell
$ pip install tensorflow==2.0.0-beta1
```
