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
from dem_sim.training import train, test, DEMLoss, VectorMetrics

# External packages dependencies
import wandb, os, h5py

if __name__ == '__main__':
    #----------- wandb configuration
    config = dict (
      batch_size = 1,
      epochs = 100,
      cutoff_distance_prefactor = 2.,
      learning_rate = 1e-6,
      num_hidden_layers = 12,
      hidden_features = 128,
      architecture = "Linear",
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
    dataset = h5py.File(data_dir + filename, 'r')
    num_samples = len([key for key in dataset.keys() if key[0].isnumeric()])
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    sample_dataset_train = SampleDataset(dataset, num_samples=(0, train_size))
    sample_dataset_val = SampleDataset(dataset, num_samples=(train_size, train_size + val_size))
    step_dataset_train = StepDataset(sample_dataset_train)
    step_dataset_val = StepDataset(sample_dataset_val)
    
    g = torch.Generator().manual_seed(0)
    loader_train = DataLoader(step_dataset_train, batch_size=config['batch_size'],
                        pin_memory=True, generator=g)
    loader_val = DataLoader(step_dataset_val, batch_size = config['batch_size'],
                        pin_memory=True, generator=g)

    #---------- Model creation
    generator = GraphGenerator(cutoff_distance=config['cutoff_distance_prefactor'] * sample_dataset_train.max_particle_radius)
    model = GNNModel(device=device, num_hidden_layers = config['num_hidden_layers'],
                    hidden_features=config['hidden_features'])
    wandb.watch(model) #This enables log pytorch gradients
    model.to(device)
    simulator = Simulator(model=model, graph_generator=generator)
    simulator.to(device)

    #---------- Optimizer and loss function initialization
    optimizer = Adam(simulator.parameters(), lr=config['learning_rate'])
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
    losses = train(simulator, optimizer, loader_train, loss_function, metric, device,
                   epochs=config['epochs'],
                   start_epoch=start_epoch,
                   start_step=start_step,
                   total_loss=previous_loss)
    
    #---------- Rollout
    """
    gd_sample = sample_dataset[10].copy_to(device)
    domain_sequence = gd_sample.domain
    time_sequence = gd_sample.time
    max_steps = 10
    predictions = simulator.rollout(gd_sample, domain_sequence[:max_steps], time_sequence[:max_steps])
    """

    #---------- Testing
    sample_dataset_test = SampleDataset(dataset, num_samples=(train_size + val_size, num_samples))
    step_dataset_test = StepDataset(sample_dataset_test)
    #loader_test = DataLoader(step_dataset_test, batch_size = config['batch_size'],
                       # pin_memory = True, generator = g)
    #test(simulator, loader_test, loss_function, device)
    