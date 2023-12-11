# kaist-sr-hypergraph

This repository describes how to run code for performing: 
- Task 1: Node disambiguation
- Task 2: Local clustering

## Step 1: Preparation

### Datasets

All required datasets are located in the below link:

https://www.dropbox.com/sh/qhz6rqol5mue4wc/AAByciJuLNba8uGv8MdMpbf-a?dl=0

### Packages

The following version (or later) of packages is required
```
numpy == 1.21.2
torch == 1.12.1
torch_geometric == 2.2.0
sklearn == 1.0.2
```

## Step 2: Run code

### Set-up

The set-up hierarchy should be as below.
Importantly, datasets provided under the hierarchy of ```temporal_mag``` and ```temporal_dblp``` (files in Dropbox) should all be located as the below hierarchy.

```
\temporal_dblp
  |_ temporal_dblp_orig_X.pickle
  |_ temporal_dblp_orig_Y.pickle
  ...
\temporal_mag
  |_ temporal_mag_orig_X.pickle
  |_ temporal_mag_orig_Y.pickle
  ...
main.py
utils.py
models.py
ssl_tools.py
```

### Running

Results can be obtained by running ```main.py``` code.

```
python3 main.py -data temporal_dblp -task task1 -device cuda:0 -epoch 10 -lr 0.001 -mask_rate 0.3
```

The meaning of each element is as follows:

```
data (str): Dataset of interest. It should be given either 'temporal_dblp' (small size) or 'temporal_mag' (large size).
task (str): Downstream task of interest. It should be given either 'task1' (node-disambiguation) or 'task2' (local-clustering).
device (str): GPU machine a user wants to use. It can be 'cuda:x' (e.g., cuda:0).
epoch (int): Self-supervised learning epochs. It can be any positive integer (e.g., 10).
lr (float): Self-supervised learning rate (step size). It can be any positive float (e.g., 10).
mask_rate (float): Self-supervised learning mask ratio. It can be any positive float between 0 and 1 (e.g., 0.2).
```
By running the above code, evaluation results for each seed and time stamps are printed
```
0 32 0.8605726062560689
```
This result implies at 0-seed and 32-timestamp, the evaluation result is 0.8606 AUROC score.
