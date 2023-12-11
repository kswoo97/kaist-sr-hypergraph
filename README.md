# kaist-sr-hypergraph

This repository describes how to run code for performing: 
- Task 1: Node disambiguation
- Task 2: Local clustering

## Step 1: Data preparation

All required datasets are located in the below link:

https://www.dropbox.com/sh/qhz6rqol5mue4wc/AAByciJuLNba8uGv8MdMpbf-a?dl=0

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
python3 main.py 
```
