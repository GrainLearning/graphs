import os.path as osp
from tqdm import tqdm
from typing import List, Tuple


import h5py
import numpy as np
import torch
from torch import nn
import torch_geometric as tg
from torch.utils.data import Dataset
from torch_cluster import radius_graph


class StepDataset(tg.data.Dataset):
    """
    Step level dataset, returns graphs.
    """
    def __init__(self, sample_dataset, loop: bool = False):
        super().__init__()
        self.sample_dataset = sample_dataset
        self.num_steps = np.sum(sample_dataset.step_counts)
        self.cumulative_steps = np.cumsum(sample_dataset.step_counts)
        self.graph_generator = GraphGenerator(sample_dataset.radius, loop)

    def len(self):
        return self.num_steps

    def get(self, idx):
        sample_idx = np.argmax(idx < self.cumulative_steps)
        if sample_idx > 0:
            step = idx % self.cumulative_steps[sample_idx - 1]
        else:
            step = idx

        return DEMGraph(self.sample_dataset.radius, step, *self.sample_dataset[sample_idx])
    # return self.graph_generator.build_graph(step, *self.sample_dataset[sample_idx])


class SampleDataset(Dataset):
    """
    Sample level dataset.
    """
    def __init__(self, path):
        super().__init__()

        self.file = h5py.File(path, 'r')
        self.output_tensor_keys = [
                'node_features',
                'radius',
                'time',
                'macro_input_features',
                'macro_output_features',
                'sample_properties',
                ]

        self.radius = 2 * (self.file['metadata/mean_radius'][()] + \
                1/2 * self.file['metadata/dispersion_radius'][()])
        self.sample_keys = [key for key in self.file.keys() if key[0].isnumeric()]
        # only count steps with a label (so with a next step)
        self.step_counts = [int(self.file[sample_key]['num_steps'][()]) - 1
                                for sample_key in self.sample_keys]

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx: int) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get full time series of sample.

        Args:
            idx (int): sample index.
        Returns:
            torch.Tensor: Node features (dynamical), shape [T, N, 9].
            torch.Tensor: Node properties (static, just the radius), shape [N, 1].
            torch.Tensor: Time deltas (dynamical), shape [T, 1].
            torch.Tensor: Domains (dynamical), shape [T, 3].
            torch.Tensor: Macroscopic features (dynamical), shape [T, 3].
            torch.Tensor: Sample properties (static), shape [4].
        """
        sample = self.file[self.sample_keys[idx]]
        outputs = {key: get_tensor(sample[key]) for key in self.output_tensor_keys}
        outputs['time'] = outputs['time'][:, :1]  # only take time delta, not step count
        return tuple(outputs.values())


class GraphGenerator(nn.Module):
    def __init__(self, radius: float, loop: bool):
        self.radius = radius
        self.loop = loop
        super().__init__()

    def build_graph(self, step: int,
                node_features: torch.Tensor,
                node_properties: torch.Tensor,
                time: torch.Tensor,
                domains: torch.Tensor,
                graph_outputs: torch.Tensor,
                graph_properties: torch.Tensor,
            ):
        """
        Create a single graph using time sequence data.

        Args:
            step (int): which timestep to take.
            node_features ([T, N, F] tensor): dynamical node features.
            node_properties ([N, F'] tensor): constant node features.
            time ([T, 1] tensor): time deltas.
            domain ([T, 3] tensor): dynamical domain size.
            graph_outputs ([T, F''] tensor): graph level dynamical outputs.
            graph_properties ([F'''] tensor): constant graph-level features.
        """
        positions = node_features[step, :, :3]
        velocities = node_features[step, :, 3:]
        next_node_features = node_features[step + 1]
        t, t_next = time[step], time[step + 1]
        domain, domain_next = domains[step], domains[step + 1]
        current_outputs = graph_outputs[step]
        graph_properties = graph_properties

        edge_index = self._compute_edges(positions, domain)

        return tg.data.Data(
            pos=positions,
            r=node_properties,
            v=velocities,
            y=next_node_features,
            t=t,
            t_next=t_next,
            domain=domain,
            domain_next=domain_next,
            y_global=current_outputs,
            x_global=graph_properties,
            edge_index=edge_index,
        )

    def _compute_edges(self, positions: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """
        Compute the radius graph, taking into account periodic boundary conditions.

        This is done by applying `radius_graph` to the original positions and thrice more
        to the positions shifted and transformed back onto the domain.

        Args:
            positions (torch.Tensor, shape [N, 3]): all node positions.
            domain (torch.Tensor, shape [3]): The domain size.

        Returns:
            torch.Tensor, shape [2, E], edge list.
        """
        edge_index = radius_graph(positions, self.radius, loop=self.loop)

        for factor in [0.25, 0.5, 0.75]:
            shift_vector = factor * domain
            positions_shifted = torch.remainder(positions + shift_vector, domain)

            edge_index_shifted = radius_graph(positions_shifted, self.radius, loop=self.loop)

            edge_index = torch.unique(torch.concat((edge_index, edge_index_shifted), dim=1), dim=1)

        return edge_index

    def evolve(self, graph, pred_x, pred_v, t_next, domain_next) -> None:
        """
        Using GNN predictions and new system inputs, evolve the graph.
        Modifies the input graph in place.

        Args:
            graph (torch_geometric.data.Data): the initial graph.
            pred_x (torch.Tensor): predictions of new positions.
            pred_v (torch.Tensor): predictions of new velocities.
            t_next (torch.Tensor): new time difference.
            domain_next (torch.Tensor): new domain size.
        """
        # NOTE: this is meant for time evolution, and so does not update
        # node or graph features, as these are not used in the evolution.
        graph.pos = pred_x
        graph.v = pred_v
        graph.t = graph.t_next
        graph.domain = graph.domain_next
        graph.t_next = t_next
        graph.domain_next = domain_next
        graph.edge_index = self._compute_edges(graph.pos)

class DEMGraph(tg.data.Data):
    """
    Subclass of `pytorch_geometric.data.Data` representing a graph in DEM.

    Attributes:
        radius: The distance below which to connect two nodes.
        pos: The positions of each nodes, shape [N, 3].
        r: The radii (or generally constant properties), shape [N, 1].
        v: The velocities and angular velocities, shape [N, 6].
        y: The positions and velocities at the next timestep, shape [N, 9].
        t: The time passed since the last step, shape [1].
        t_next: As above for next timestep.
        domain: The domain size, shape [3].
        domain_next: As above for the next timestep.
        y_global: The macroscopic properties, shape [3].
        x_global: The constant graph properties.
        edge_index: The edges as found by.
    """
    def __init__(self, radius: float, step: int,
        node_features: torch.Tensor,
        node_properties: torch.Tensor,
        time: torch.Tensor,
        domains: torch.Tensor,
        graph_outputs: torch.Tensor,
        graph_properties: torch.Tensor,
        loop: bool = False,
        ):
        """
        Initialize a DEMGraph, using time sequence data.

        Args:
            step (int): which timestep to take.
            radius (float): cutoff distance below which to connect nodes.
            loop (bool, default False): whether or not to include self loops.
            node_features ([T, N, F] tensor): dynamical node features.
            node_properties ([N, F'] tensor): constant node features.
            time ([T, 1] tensor): time deltas.
            domain ([T, 3] tensor): dynamical domain size.
            graph_outputs ([T, F''] tensor): graph level dynamical outputs.
            graph_properties ([F'''] tensor): constant graph-level features.
        """
        positions = node_features[step, :, :3]
        velocities = node_features[step, :, 3:]
        next_node_features = node_features[step + 1]
        t, t_next = time[step], time[step + 1]
        domain, domain_next = domains[step], domains[step + 1]
        current_outputs = graph_outputs[step]
        graph_properties = graph_properties

        super().__init__(
            pos=positions,
            r=node_properties,
            v=velocities,
            y=next_node_features,
            t=t,
            t_next=t_next,
            domain=domain,
            domain_next=domain_next,
            y_global=current_outputs,
            x_global=graph_properties,
            radius=radius,
            loop=loop,
        )

        self.edge_index = self._compute_edges()

    def evolve(self, pred_x, pred_v, t_next, domain_next):
        """
        Using GNN predictions and new system inputs, evolve the graph.

        Args:
            pred_x (torch.Tensor): predictions of new positions.
            pred_v (torch.Tensor): predictions of new velocities.
            t_next (torch.Tensor): new time difference.
            domain_next (torch.Tensor): new domain size.
        """
        # NOTE: this is meant for time evolution, and so does not update
        # node or graph features, as these are not used in the evolution.
        self.pos = pred_x
        self.v = pred_v
        self.t = self.t_next
        self.domain = self.domain_next
        self.t_next = t_next
        self.domain_next = domain_next
        self.edge_index = self._compute_edges()

    def _compute_edges(self):
        """
        Compute the radius graph, taking into account periodic boundary conditions.

        This is done by applying `radius_graph` to the original positions and thrice more
        to the positions shifted and transformed back onto the domain.
        """
        edge_index = radius_graph(self.pos, self.radius, loop=self.loop)

        for factor in [0.25, 0.5, 0.75]:
            shift_vector = factor * self.domain
            positions_shifted = torch.remainder(self.pos + shift_vector, self.domain)

            edge_index_shifted = radius_graph(positions_shifted, self.radius, loop=self.loop)

            edge_index = torch.unique(torch.concat((edge_index, edge_index_shifted), dim=1), dim=1)

        return edge_index


def get_tensor(data) -> torch.Tensor:
    """Convert data inside an hdf5 file to a torch tensor."""
    return torch.tensor(data[()])


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

