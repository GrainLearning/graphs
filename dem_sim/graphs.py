import os.path as osp
from tqdm import tqdm
from typing import List


import h5py
import numpy as np
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset
from torch_cluster import radius_graph


class Dataset1Step(Dataset):
    """
    Save all graphs separately to disk.
    Takes very long (10 hours on CPU), and increases file size by factor of 4,
    for a total of 100Gb.
    """
    def __init__(
        self,
        root,
        check_edges: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.check_edges = check_edges
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'simState_path_sampling_5000_graphs.hdf5'

    @property
    def processed_file_names(self):
        return 'simState_path_sampling_5000_graphs_processed.pt'

    def download(self):
        pass

    def process(self):
        self.data = h5py.File(self.raw_paths[0], 'r')

        max_radius = np.array(self.data['metadata']['mean_radius']) + \
                1/2 * np.array(self.data['metadata']['dispersion_radius'])
        sample_keys = [key for key in self.data.keys() if key[0].isnumeric()]
        idx = 0
        for sample_key in tqdm(sample_keys):
            sample = self.data[sample_key]
            final_step = max([int(s) for s in list(sample['time_sequence'].keys())])

            for step in range(final_step):
            # purposefully excluding the last step as it has no labels
                data = build_graph(sample, step, radius=2 * max_radius)

                # this is the bottleneck, so only check now and then
                if self.check_edges and step % 100 == 0:
                    gnn_edges = data.edge_index
                    dem_edges = torch.stack((
                        get_tensor(sample[f'time_sequence/{step}/sources']),
                        get_tensor(sample[f'time_sequence/{step}/destinations']))
                    )
                    missing_edges = find_missing(dem_edges, gnn_edges)
                    if missing_edges:
                        print(f"In sample {sample_key}, timestep {step}, "
                              f"{len(missing_edges)} DEM edges were missed.")

                torch.save(data, self.graph_name(idx))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.graph_name(idx))
        return data

    def graph_name(self, idx):
        return osp.join(self.processed_dir, f'data_{idx}.pt')


def build_graph(sample, t: int, radius: float) -> tg.data.Data:
    """
    Convert data in hdf5 file to a torch geometric Data object.

    Args:
        sample (hdf5 group): Group containing all data of a specific sample.
        t (int): The timestep at which to extract the data.
        radius (float): Threshold distance at which to connect two nodes.

    Returns:
        A graph with the properties:
            pos: The positions of each nodes, shape [N, 3].
            r: The radii (or generally constant properties), shape [N, 1].
            v: The velocities and angular velocities, shape [N, 6].
            y: The positions and velocities at the next timestep, shape [N, 9].
            t: The current timestep and ..., shape [2]. TODO what is this exactly?
            t_next: As above for next timestep.
            domain: The domain size, shape [3].
            domain_next: As above for the next timestep.
            y_global: The macroscopic properties, shape [3].
            edge_index: The edges as found by `compute_edges`.
    """

    current_step, next_step = sample[f'time_sequence/{t}'], sample[f'time_sequence/{t + 1}']

    node_properties = get_tensor(sample['metadata/radius']).float()  # otherwise 64 bit
    node_features = get_tensor(current_step['node_features'])

    # The first 3 node features are the position, which we can provide separately
    positions = node_features[:, :3]
    velocities = node_features[:, 3:]

    next_node_features = get_tensor(next_step['node_features'])

    # graph-level features
    domain = get_tensor(current_step['macro_input_features'])
    domain_next = get_tensor(next_step['macro_input_features'])
    t, t_next = get_tensor(current_step['time']), get_tensor(next_step['time'])

    current_outputs = get_tensor(current_step['macro_output_features'])

    # computing edges
    edge_index = compute_edges(positions, domain, radius, loop=False)

    graph = tg.data.Data(
        pos=positions,
        r=node_properties,
        v=velocities,
        y=next_node_features,
        edge_index=edge_index,
        t=t,
        t_next=t_next,
        domain=domain,
        domain_next=domain_next,
        y_global=current_outputs,
    )

    return graph


def compute_edges(positions, domain, radius, **kwargs):
    """
    Compute the radius graph, taking into account periodic boundary conditions.

    This is done by applying `radius_graph` to the original positions and twice more
    to the positions shifted and transformed back onto the domain.

    Args:
        positions (torch.Tensor): Node positions, shape [N, 3].
        domain (torch.Tensor): Domain size, shape [3].
        radius (float): Threshold radius below which distance to connect nodes.
        kwargs: Any keyword arguments for `radius_graph`.
    Returns:
        edge index (torch.Tensor), shape [2, E].
    """
    edge_index = radius_graph(positions, radius, **kwargs)

    for factor in [0.4, 0.6]:
        shift_vector = factor * domain
        positions_shifted = torch.remainder(positions + shift_vector, domain)

        edge_index_shifted = radius_graph(positions_shifted, radius, **kwargs)

        edge_index = torch.unique(torch.concat((edge_index, edge_index_shifted), dim=1), dim=1)

    return edge_index


def get_tensor(data) -> torch.Tensor:
    """Convert data inside an hdf5 file to a torch tensor."""
    return torch.tensor(data[:])


def find_missing(A: torch.Tensor, B: torch.Tensor) -> List[tuple]:
    """
    Return edges from edge list A that are missing in edge list B.

    Args:
        A (torch.Tensor): edge list, shape [2, E].
        B (torch.Tensor): edge list, shape [2, E'].

    Returns:
        list of tuples
    """
    A_list = [tuple(edge.numpy()) for edge in list(A.T)]
    B_set = {tuple(edge.numpy()) for edge in list(B.T)}

    A_not_in_B = [edge for edge in A_list if edge not in B_set]
    return A_not_in_B
