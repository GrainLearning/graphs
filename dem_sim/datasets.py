import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dem_sim.data_types import GraphData


class StepDataset(Dataset):
    """
    Step level dataset. Samples all steps of all samples exactly once.

    Returns the step number and all the sample data.
    """
    def __init__(self, sample_dataset, loop: bool = False):
        super().__init__()
        self.sample_dataset = sample_dataset
        self.num_steps = np.sum(sample_dataset.step_counts)
        self.cumulative_steps = np.cumsum(sample_dataset.step_counts)

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx) -> (int, GraphData):
        sample_idx = np.argmax(idx < self.cumulative_steps)
        if sample_idx > 0:
            step = idx % self.cumulative_steps[sample_idx - 1]
        else:
            step = idx

        return self.sample_dataset[sample_idx], step


class SampleDataset(Dataset):
    """
    Sample level dataset.
    """
    def __init__(self, path):
        super().__init__()
        self.file = h5py.File(path, 'r')

        self.max_particle_radius = self.file['metadata/radius_max'][()]
        self.sample_keys = [key for key in self.file.keys() if key[0].isnumeric()]
        # only count steps with a label (so with a next step)
        self.step_counts = [int(self.file[sample_key]['num_steps'][()]) - 1
                                for sample_key in self.sample_keys]

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx: int) -> GraphData:
        """
        Get full time series of sample.

        Args:
            idx (int): sample index.
        Returns:
            GraphData
        """
        sample = self.file[self.sample_keys[idx]]
        return GraphData(**{key: get_tensor(sample[key]) for key in GraphData._fields})


def get_tensor(data) -> torch.Tensor:
    """Convert data inside an hdf5 file to a torch tensor."""
    return torch.tensor(data[()])

