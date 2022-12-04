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
from dem_sim.training import train, DEMLoss, VectorMetrics

# External packages dependencies
import wandb, os

if __name__ == '__main__':
    #----------- wandb configuration
    config = dict (
      batch_size = 1,
      epochs = 10,
      cutoff_distance_prefactor = 2.,
      architecture = "CNN",
      dataset_id = "path_sampling_5000",
      infrastructure = "crib utwente",
      device = "cuda"
    )

    wandb.init(project = "GrainLearning_GNN_1",
        entity = "grainlearning-escience",
        notes = "testing time performance",
        tags = ["baseline", "paper1"],
        resume = True, # True: resume the run next time (must be in the same machine)
        config = config)

    #---------- Simulator configuration
    if config['infrastructure'] == "Snellius":
        data_dir = '/projects/0/einf3381/GrainLearning/TrainingData/PathSampling/'
    elif config['infrastructure'] == "crib utwente":
        data_dir = '/data/private/'
    elif config['infrastructure'] == "MacOS Luisa":
        data_dir = '/Users/luisaorozco/Documents/Projects/GrainLearning/data/gnn_data/'
    elif config['infrastructure'] == "MacOS Aron":
        data_dir = '/Users/aronjansen/Documents/grainsData/raw/'

    #---------- Data loading
    device = torch.device(config['device'])
    filename = 'simState_path_sampling_5000_graphs_reformatted.hdf5'
    sample_dataset = SampleDataset(data_dir + filename)
    step_dataset = StepDataset(sample_dataset)
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(step_dataset, batch_size = config['batch_size'],
                        pin_memory = True, generator = g)

    #---------- Model creation
    generator = GraphGenerator(cutoff_distance = config['cutoff_distance_prefactor'] * sample_dataset.max_particle_radius)
    model = GNNModel(device = device)
    wandb.watch(model) #This enables log pytorch gradients
    model.to(device)
    simulator = Simulator(model = model, graph_generator = generator)
    simulator.to(device)

    #---------- Optimizer and loss function initialization
    optimizer = Adam(simulator.parameters())
    loss_function = DEMLoss()
    metric = VectorMetrics()

    #---------- Loading a checkpoint
    if not os.path.isdir("outputs"): os.mkdir("outputs")
    start_step, start_epoch= 0, 0
    previous_loss = 0.0
    if os.path.isfile('outputs/model.pth'):
        checkpoint = torch.load('outputs/model.pth')
        simulator.load_state_dict(checkpoint['model_state_dict'])
        print('Previously trained model weights state_dict loaded...')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Previously trained optimizer state_dict loaded...')
        metric.load_state_dict(checkpoint['metric_state_dict'])
        print('Previously trained metric state_dict loaded...')
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        print(f"Previously trained for {start_epoch} epochs, and {start_step} steps ...")
        previous_loss = checkpoint['total_loss_epoch']
    
    #---------- Training
    losses = train(simulator, optimizer, loader, loss_function, metric, device,
                   epochs = config['epochs'],
                   start_epoch = start_epoch,
                   start_step = start_step,
                   total_loss = previous_loss)
    
    #---------- Rollout
    """
    gd_sample = sample_dataset[10].copy_to(device)
    domain_sequence = gd_sample.domain
    time_sequence = gd_sample.time
    max_steps = 10
    predictions = simulator.rollout(gd_sample, domain_sequence[:max_steps], time_sequence[:max_steps])
    """
