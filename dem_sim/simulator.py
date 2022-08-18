from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn import InstanceNorm, MessagePassing, global_mean_pool


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
        ):
        super().__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        # 7 is original node features (without position)
        message_inputs = in_features + graph_features + spatial_dimension + \
                            2 * (velocity_dimension + property_dimension)
        self.message_net = nn.Sequential(
            nn.Linear(message_inputs, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )

        update_inputs = in_features + hidden_features + graph_features
        self.update_net = nn.Sequential(
            nn.Linear(update_inputs, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.ReLU(),
        )

        self.norm = InstanceNorm(hidden_features)

    def forward(self, h, v, pos, r, graph_features, edge_index, batch):
        """
        Propagate messages along edges, following Brandstetter.
        Args:
            h: Hidden features, shape [N, hidden_features].
            v: Node velocities, shape [N, 6].
            pos: Node positions, shape [N, 3].
            r: Node radii, shape [N, 1].
            graph_features: input graph-level features
            edge_index: edge index
            batch: tensor indicating which nodes belong to which graph
        """
        h = self.propagate(edge_index, h=h, v=v, pos=pos, r=r,
                graph_features=graph_features)
        h = self.norm(h, batch)
        return h

    def message(self, h_i, h_j, v_i, v_j, pos_i, pos_j, r_i, r_j, graph_features):
        graph_features = graph_features.repeat(h_i.shape[0], 1)
        message_input = torch.cat(
                (h_i - h_j, v_i, v_j, r_i, r_j, pos_i - pos_j, graph_features),
                dim=1)

        message = self.message_net(message_input)
        return message

    def update(self, message, h, graph_features):
        graph_features = graph_features.repeat(h.shape[0], 1)

        update_input = torch.cat((h, message, graph_features), dim=1)
        update = self.update_net(update_input)
        if self.in_features == self.out_features:
            return h + update
        else:
            return update


class Simulator(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 6,
        hidden_features: int = 128,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.output_features = 3 + 6  # position, velocities
        self.output_macro_features = 4  # really?
        self.graph_features = 2 * 5  # domain and time, this and next step
        self.input_features = self.graph_features + 1 + 6  # radius, velocity, NOTE: removed position here

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
        )

        self.gnn_layers = nn.ModuleList(modules=(
            GNN_Layer(
                in_features=self.hidden_features,
                hidden_features=self.hidden_features,
                out_features=self.hidden_features,
                graph_features=self.graph_features,
                )
            for _ in range(self.num_hidden_layers))
            )
        # Brandstetter adds the last one separately, but with identical arguments?

        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.output_features),
        )

        self.macro_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.output_macro_features),
        )

    def forward(self, graph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v, pos, r = graph.v, graph.pos, graph.r
        edge_index = graph.edge_index
        batch = graph.batch

        # use the current domain size and timestep and the increment as globals
        # TODO: add other input graph level features from metadata
        graph_features = torch.cat((graph.domain, graph.t))  # TODO: maybe only take first of graph.t
        graph_features_next = torch.cat((graph.domain_next, graph.t_next))
        graph_features = torch.cat((graph_features, graph_features_next))

        node_features = torch.cat((graph.pos, graph.v), dim=1)
        all_features = torch.cat((
            graph.r,
            node_features,
            graph_features.repeat(node_features.shape[0], 1)
            ), dim=1)

        features_without_position = torch.cat((
            graph.r,
            graph.v,
            graph_features.repeat(node_features.shape[0], 1),
            ), dim=1)
        h = self.embedding_mlp(features_without_position)

        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, v, pos, r, graph_features, edge_index, batch)

        prediction = self.output_mlp(h) + node_features

        pred_x = torch.remainder(prediction[:, :3], graph.domain_next)
        pred_v = prediction[:, 3:]

        h_mean = global_mean_pool(h, batch)
        pred_macro = self.macro_mlp(h_mean)[0]  # get rid of empty node dimension

        return pred_x, pred_v, pred_macro

    def rollout(self, g_0, sequence):
        pass

