from typing import Callable

import torch
from torch import nn
from torch_geometric.nn import InstanceNorm, MessagePassing, global_mean_pool
from dem_sim.utils import periodic_difference


class GNN_Layer(MessagePassing):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            graph_features: int,
            spatial_dimension: int = 3,
            velocity_dimension: int = 6,
            property_dimension: int = 1,
            normalize: bool = False,
            activation: Callable = nn.ReLU,
        ):
        super().__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        message_inputs = 2 * in_features + graph_features + spatial_dimension + \
                            2 * (velocity_dimension + property_dimension)
        self.message_net = nn.Sequential(
            nn.Linear(message_inputs, hidden_features),
            activation(),
            nn.Linear(hidden_features, hidden_features),
            activation(),
        )

        update_inputs = in_features + hidden_features + graph_features
        self.update_net = nn.Sequential(
            nn.Linear(update_inputs, hidden_features),
            activation(),
            nn.Linear(hidden_features, out_features),
            activation(),
        )

        self.normalize = normalize
        self.norm = InstanceNorm(hidden_features)

    def forward(self, h, v, w, pos, r, graph_features, domain, edge_index, batch):
        """
        Propagate messages along edges, following Brandstetter.
        Args:
            h: Hidden features, shape [N, hidden_features].
            v: Node velocities, shape [N, 3].
            w: Node angular velocities, shape [N, 3].
            pos: Node positions, shape [N, 3].
            r: Node radii, shape [N, 1].
            graph_features: input graph-level features
            domain: Domain size, shape [3].
            edge_index: edge index
            batch: tensor indicating which nodes belong to which graph
        """
        h = self.propagate(edge_index, h=h, v=v, w=w, pos=pos, r=r,
                graph_features=graph_features, domain=domain)
        if self.normalize:
            h = self.norm(h, batch)
        return h

    def message(self, h_i, h_j, v_i, v_j, w_i, w_j, pos_i, pos_j, r_i, r_j, graph_features, domain):
        graph_features = graph_features.repeat(h_i.shape[0], 1)
        pos_diff = periodic_difference(pos_i, pos_j, domain)
        message_input = torch.cat(
                (h_i, h_j, v_i, v_j, w_i, w_j, r_i, r_j, pos_diff, graph_features),
                dim=1)

        message = self.message_net(message_input)
        return message

    def update(self, message, h, graph_features):
        graph_features = graph_features.repeat(h.shape[0], 1)

        update_input = torch.cat((h, message, graph_features), dim=1)
        update = self.update_net(update_input)
        if self.in_features == self.out_features:
            update = update + h

        return update


class GlobalMeanPool(nn.Module):
    """
    Wrapper around global_mean_pool.
    """
    def forward(self, x, batch=None, size=None):
        return global_mean_pool(x, batch, size)


class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
