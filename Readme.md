# Companion code for Insanally, Albanna, et al. "Contributions and synaptic basis of diverse cortical neuron responses to task performance"

This is the code acccompanying "Contributions and synaptic basis of diverse cortical neuron responses to task performance" [currently available on the bioRxiv](https://www.biorxiv.org/content/10.1101/2022.05.04.490676v1)

This repo contains two seperate codebases:

1. Code for simulating task-performing, spiking neural networks with FORCE and STDP mechansms  written in [Julia](https://julialang.org/) (`simulation`)
2. and code for analyzing the results of these simulations written in [Python](https://www.python.org/) (`analysis`)

## System requirements 

This software has been tested on macOS version 11 and higher and ubuntu version 18 and higher. Simulations require at least 8 GB of 
RAM. 

### `simulation` requirements

* Julia >= 1.9

### `anaysis` requirements

* python >= 3.9
* A jupyter notebook editor (e.g. [JupyterLab](https://jupyter.org/), [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), or [nteract](https://nteract.io/))

## Installation Guide 

### `simulation` installation 

The project contains the `Project.toml` and `Manifest.toml` required to create the necessary environment to execute the code contained in `spiking_LIF_w_FORCE_STDP.jl`. 


1. [Install Juila](https://julialang.org/) 
2. Go to the directory where this repo is housed (e.g. `.../InsanallyAlbanna2023`) and enter the shell command to create the environment and install the necesssary packages:

```
~/InsanallyAlbanna2023> julia setup.jl
```  

Setup should only take a few minutes

### `analysis` installation

To install the environment required for the Jupyter notebooks containing most analyses for Figs 1-5 use the included requirements.txt. 

1. [Create a virtual python environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
2. Activate your virtual environment
3. From the shell in the repo driectory: 
```
~/InsanallyAlbanna2023> pip install -r requirements.txt
``` 
4. To create a ipython kernel for this environment in jupyter run:
```
~/InsanallyAlbanna2023> python -m ipykernel install --user --name=IA23
```
5. Select `IA23` as your kernel when running one of the Jupyter notebook editors above.

## `simulation` Demo 

To run an example simulation which will save outputs in the `example_simulations` directory enter the shell command

```
~/InsanallyAlbanna2023> julia example_simulation.jl
```

The script is short and contains the necessary commands to modify the simulation paramaters. On a 4.2 GHz Quad-Core Intel Core i7 running macOS the simulation takes ~30 minutes to complete. 

Outputs produced in `example_simulations` include 

1. Saved snapshots of network activity 
2. Plot of injected and feedback current vs trial (+ csv of data)
3. Plots of example unit voltages on individual trails & current inputs (+ csvs of data)
4. Plot of network-wide spiking activity on a trial (+ csv of data)
5. Plot of readout node activity on a trial (+ csv of data)
6. Plot of population firing rates vs trial (+ csv of data)
7. Plot of synaptic weights vs trial (+ csv of data)

