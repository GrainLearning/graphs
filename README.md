# GNN for DEM simulations

Creating and training Graph Neural Networks for predicting the mechanics of granular materials tested under triaxial compression.

## Requirements

In this implementation we are using:
- [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) for the graphs and GNN layer creation.
- [Pytorch](https://pytorch.org/) as the ML backend. Our code supports the usage of accelerators such as nvidia gpus (cuda).
- [wandb](https://wandb.ai) to track the experiments.

Some additional requirements are:

- [h5py](https://pypi.org/project/h5py/)
- [numpy](https://pypi.org/project/numpy/)
- [tqdm](https://pypi.org/project/tqdm/)
- [torch_cluster](https://github.com/rusty1s/pytorch_cluster) (Should have been installed with Pythorch Geometric)

## Example of usage

In the script [train_one_step_wandb.py](train_one_step_wandb.py) you can find an example for how to train a GNN. It consists on the following stages:

1. **Configure the simulator:** the hyper-parameters in the dictionary `config`.
1. **Configure wandb:** Specify wandb details in `wand.init`.
1. **Data loading:** We have transformed the dataset containing several DEM simulations data into a dataset: `simState_path_sampling_5000_graphs_reformatted.hdf5`, if you want to get this dataset, contact Hongyang Cheng.
1. **Model creation:** Here we create 3 different objects: `GraphGenerator, GNNModel, Simulator` [more info](#elements-of-the-trade).
1. **Optimizer and loss function initialization:**
1. **Loading a checkpoint:** Functionality to re-start/continue the training of a model (for example if your code crashes during the training). Will only be triggered if there was a model trained before (i.e. there is a file containing the model in `outputs` folder.)
1. **Training:** Call the train function with all the elements that we have created before.
1. :construction: **Rollout:** :warning: needs to be tested. Continuous prediction of consecutive time-steps, thus, each prediction is made based on a model's prediction for the previous step.

You can simply run it as `python train_one_step_wandb.py`

## Elements of the trade

- **Graph generator**
  - Builds the graph from the DEM simulation data.
  - Calculate the edges (connections between nodes or particles).
  - Updates the graph (nodes and edges values).

- **GNN Layer**

  Graph Neural network layer in charge of performing the message passing process through the graph.
  - *Message* $\phi =$ MLP($h_i - h_j, v_i, v_j, r_i, r_j, x_i - x_j \% domain, \Theta$)
  considering $i$ as the central node and $j$ all neighbors of $i$:
    - $h_i$ and $h_j$: Hidden features of nodes (particles) $i$ and $j$.
    - $v_i$ and $v_j$: velocity vector of nodes (particles) $i$ and $j$.
    - $r_i$ and $r_j$: Radius of nodes (particles) $i$ and $j$.
    - $x_i - x_j \% domain$: position vectorial difference between nodes $i$ and $j$, trimmed to the domain size. (necessary the DEM simulations have periodic boundary conditions).
    - $\Theta$: vector of graph features (not specific of each node) such as domain volume and time of current and next step, contact parameters, and sample properties: compressive strain rate $\dot{\varepsilon_z}$, initial friction, confinment pressure, shear strain rate $\dot{}\varepsilon_q$.
    - MLP: two stacked torch linear layers with RELU activation functions. These have only (trainable) weights, no biases. All previous parts are concatenates, and then passed trough the MLP.
    - The *aggregation* of the messages is trough *mean*.

  - *Update function $\gamma =$* MLP $(h, \phi, \Theta) + h$
    - These MLP is different from the one of the message passing, its weights are also trainable and it has no biases. $h, \phi, \Theta$ are concatenated and then passed to the MLP.

  - General GNN layer function:
  $h_i^k = \gamma[h_i^{k-1}, \frac{1}{\cal{N}(i)} \sum_{\cal{N}(i)} \phi(h_i^{k-1}, h_j^{k-1})]$

- **Model**
 `torch.nn.Module` that stacks layers, adds activation functions, dropouts and pools (architecture).

- **Simulator**
  
  Object that drives the simulation. In a simulation there are two steps:
  1. Building a graph (from previous step or from data).
  2. Calling the Model (a forward pass) with the graph that was built in 1.

## Help and Support

For assistance with the GrainLearning software, please raise an issue on the GitHub Issues page, or contact Hongyang Cheng.
