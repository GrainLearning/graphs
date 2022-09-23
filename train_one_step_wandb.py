# Pytorch dependencies
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss

# graphs dependencies
from dem_sim.datasets import SampleDataset, StepDataset
from dem_sim.simulator import Simulator
from dem_sim.generator import GraphGenerator
from dem_sim.model import GNNModel, NaiveForecasting
from dem_sim.training import train, DEMLoss

# External packags dependencies
import wandb

if __name__ == '__main__':
    #----------- wandb configuration ----------------
    config = dict (
      batch_size = 1,
      cutoff_distance_prefactor = 2.,
      architecture = "CNN",
      dataset_id = "path_sampling_5000",
      infrastructure = "crib utwente",
      device = "cuda"
    )

    wandb.init(project="GrainLearning_GNN_1",
        entity="grainlearning-escience",
        notes="tweak baseline",
        tags=["baseline", "paper1"],
        config=config)

    #---------- Simulator configuration
    if config['infrastructure'] == "Snellius":
        data_dir = '/projects/0/einf3381/GrainLearning/TrainingData/PathSampling/'
    elif config['infrastructure'] == "crib utwente":
        data_dir = '/data/private/'
    elif config['infrastructure'] == "MacOS Luisa":
        data_dir = '/Users/luisaorozco/Documents/Projects/GrainLearning/data/gnn_data/'
    elif config['infrastructure'] == "MacOS Aron":
        data_dir = '/Users/aronjansen/Documents/grainsData/raw/'

    filename = 'simState_path_sampling_5000_graphs_reformatted.hdf5'
    sample_dataset = SampleDataset(data_dir + filename)
    step_dataset = StepDataset(sample_dataset)
    device = torch.device(config['device'])

    generator = GraphGenerator(cutoff_distance=config['cutoff_distance_prefactor'] * sample_dataset.max_particle_radius)
    model = GNNModel(device=device)
    wandb.watch(model) #This enables log pytorch gradients
    model.to(device)
    simulator = Simulator(model=model, graph_generator=generator)
    simulator.to(device)

    optimizer = Adam(simulator.parameters())
    loader = DataLoader(step_dataset, batch_size=config['batch_size'], shuffle=True)
    loss_function = DEMLoss()

    #---------- Training
    losses = train(simulator, optimizer, loader, loss_function, device)
    #wandb.log({"losses": losses})
    
    #---------- Rollout
    """
    gd_sample = sample_dataset[10].copy_to(device)
    domain_sequence = gd_sample.domain
    time_sequence = gd_sample.time
    max_steps = 10
    predictions = simulator.rollout(gd_sample, domain_sequence[:max_steps], time_sequence[:max_steps])
    """