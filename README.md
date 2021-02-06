# Spatio-Temporal Graph Convolutional Networks for Nonverbal Language in Entrepreneurial Pitching Sessions

## Introduction

TODO: Description of the project

This code has been branched and modified from [MMSkeleton](https://github.com/open-mmlab/mmskeleton), an open source toolbox for skeleton-based human understanding and part of the [open-mmlab](https://github.com/open-mmlab) in the charge of [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

## How to use our code

### Installation

a. Clone HumanNoVeLa dataset.

	    git clone https://bitbucket.org/fserracant/humannovela.git

b. Clone this repository.

	    git clone https://bitbucket.org/fserracant/stgcnnonverbal.git
	
c. Get into STGCNNonVerbal repository.
	
       cd stgcnnonverbal
		
d. Follow installation steps in [Installation](./doc/GETTING_STARTED.md) and activate open-mmlab environment.

### Creating Experiments

a. Create an experiment from the HumanNoVeLa dataset. An experiment is a folder with a data set is prepared for training and testing purposes. `prepareExperiment.py` script will perpare the HumanNoVeLa dataset for our network and partition it into training and testing sets. It uses template yaml files for training and test that can be found at the templates directory. Run `python prepareExperiment.py -h` for full list of optional arguments.
	
	    cd experiments
	    python prepareExperiment.py --dataset ../../humannovela/entrepreneurs/dataset.pkl --exppath <your_experiment_name>
	
b. (Optional) Change `train.yaml` or `test.yaml` to your needs.

c. (Optional) Change `checkpoint` file in `argparse_cfg` and `processor_cfg` in `test.yaml` file to `../checkpoints/st_gcn.HumanNoVeLa.pth` in order to use our trained weights.

### Training and testing

a. Run training on your new experiment.

        python ../mmskl.py <your_experiment_name>/train.yaml
        
b. Run testing on your experiment.

        python ../mmskl.py <your_experiment_name>/test.yaml
        
c. Find your checkpoints, logs and results in your experiment directory.

## Contact
For any question, feel free to contact
```
Joan Francesc Serracant     : francesc.serracant@gmail.com
Coloma Ballester            : coloma.ballester@upf.edu
```
