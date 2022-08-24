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
            normalize: bool = False,
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

        self.normalize = normalize
        self.norm = InstanceNorm(hidden_features)

    def forward(self, h, v, pos, r, graph_features, domain, edge_index, batch):
        """
        Propagate messages along edges, following Brandstetter.
        Args:
            h: Hidden features, shape [N, hidden_features].
            v: Node velocities, shape [N, 6].
            pos: Node positions, shape [N, 3].
            r: Node radii, shape [N, 1].
            graph_features: input graph-level features
            domain: Domain size, shape [3].
            edge_index: edge index
            batch: tensor indicating which nodes belong to which graph
        """
        h = self.propagate(edge_index, h=h, v=v, pos=pos, r=r,
                graph_features=graph_features, domain=domain)
        if self.normalize:
            h = self.norm(h, batch)
        return h

    def message(self, h_i, h_j, v_i, v_j, pos_i, pos_j, r_i, r_j, graph_features, domain):
        graph_features = graph_features.repeat(h_i.shape[0], 1)
        pos_diff = periodic_difference(pos_i, pos_j, domain)
        message_input = torch.cat(
                (h_i - h_j, v_i, v_j, r_i, r_j, pos_diff, graph_features),
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


def periodic_difference(x_i, x_j, domain):
    """
    Compute x_i - x_j taking into account the periodic boundary conditions.
    """
    diff = x_i - x_j
    smaller_one = x_i < x_j  # component-wise check which is bigger
    domain_shift = (1 - 2 * smaller_one) * domain
    diff_shifted = diff - domain_shift
    # boolean indicating in which component to use the original difference
    use_original = torch.abs(diff) < torch.abs(diff_shifted)
    periodic_diff = use_original * diff + ~use_original * diff_shifted
    return periodic_diff


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
        self.output_macro_features = 3
        self.graph_features = 2 * 4 + 4  # domain and time, this and next step, plus 4 global constant properties
        self.input_features = self.graph_features + 1 + 6  # radius, velocity

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
                normalize=True,
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
        domain = graph.domain
        edge_index = graph.edge_index
        batch = graph.batch

        # use the current domain size and timestep and the increment as globals
        graph_features = torch.cat((graph.domain, graph.t, graph.x_global))
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

        for i, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(h, v, pos, r, graph_features, domain, edge_index, batch)

        prediction = self.output_mlp(h) + node_features

        pred_x = torch.remainder(prediction[:, :3], graph.domain_next)
        pred_v = prediction[:, 3:]

        h_mean = global_mean_pool(h, batch)
        pred_macro = self.macro_mlp(h_mean)[0]  # gets rid of empty node dimension

        return pred_x, pred_v, pred_macro

    def rollout(self, g_0, domain_sequence, t_sequence):
        g = g_0
        T = t_sequence.shape[0]
        preds_x, preds_v, preds_macro = [], [], []

        for t in range(T - 1):  # TODO: think about range ends
            print(t)
            pred_x, pred_v, pred_macro = self(g)
            preds_x.append(pred_x.detach())
            preds_v.append(pred_v.detach())
            preds_macro.append(pred_macro.detach())

            g.evolve(pred_x, pred_v, t_sequence[t + 1], domain_sequence[t + 1])

        preds_x = torch.stack(preds_x)
        preds_v = torch.stack(preds_v)
        preds_macro = torch.stack(preds_macro)

        return preds_x, preds_v, preds_macro

