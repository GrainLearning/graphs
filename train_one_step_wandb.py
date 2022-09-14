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
	  learning_rate = 0.01,
	  batch_size = 1,
	  cutoff_distance_prefactor = 2.,
	  architecture = "CNN",
	  dataset_id = "peds-0192",
	  infrastructure = "Snellius",
	  device = "cuda"
	)

	wandb.init(project="GrainLearning_GNN_1",
		notes="tweak baseline",
		tags=["baseline", "paper1"],
	  	config=config)

	#---------- Simulator configuration
	data_dir = '/projects/0/einf3381/GrainLearning/TrainingData/PathSampling/'
	filename = 'simState_path_sampling_5000_graphs_reformatted.hdf5'
	sample_dataset = SampleDataset(data_dir + filename)
	step_dataset = StepDataset(sample_dataset)

	generator = GraphGenerator(cutoff_distance=config['cutoff_distance_prefactor'] * sample_dataset.max_particle_radius)
	model = GNNModel()
	wandb.watch(model) #This enables log pytorch gradients
	simulator = Simulator(model=model, graph_generator=generator)

	optimizer = Adam(simulator.parameters())
	loader = DataLoader(step_dataset, batch_size=config['batch_size'], shuffle=True)
	loss_function = DEMLoss()
	device = torch.device(config['device'])

	losses = train(simulator, optimizer, loader, loss_function, device)
	wandb.log({"losses": losses})
