from typing import Callable

import torch
from torch import nn

from dem_sim.layers import GNN_Layer, GlobalMeanPool
from dem_sim.data_types import Prediction, Graph


class GNNModel(nn.Module):
    def __init__(
        self,
        device,
        num_hidden_layers: int = 6,
        hidden_features: int = 128,
        activation: Callable = nn.ReLU,
        scale_position: float = 1e-3,
        scale_velocity: float = 1e-3,
        scale_angular_velocity: float = 1e-2,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.output_features = 3 + 6  # position, velocities
        self.output_macro_features = 3
        self.graph_features = 2 * 4 + 4  # domain and time, this and next step, plus 4 global constant properties
        self.input_features = self.graph_features + 1 + 6  # radius, velocity
        self.scale_position = scale_position,
        self.scale_velocity = scale_velocity,
        self.scale_angular_velocity = scale_angular_velocity,

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_features),
            activation(),
            nn.Linear(self.hidden_features, self.hidden_features),
            activation(),
        )

        self.gnn_layers = nn.ModuleList(modules=(
            GNN_Layer(
                in_features=self.hidden_features,
                hidden_features=self.hidden_features,
                out_features=self.hidden_features,
                graph_features=self.graph_features,
                normalize=True,
                activation=activation,
                )
            for _ in range(self.num_hidden_layers))
            )
        # Brandstetter adds the last one separately, but with identical arguments?

        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            activation(),
            nn.Linear(self.hidden_features, self.output_features),
        )

        self.pool = GlobalMeanPool()
        self.macro_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            activation(),
            nn.Linear(self.hidden_features, self.output_macro_features),
        )

        self.device = device
    def forward(self, graph: Graph, detach: bool = False) -> Prediction:
        v, w, pos, r = graph.v, graph.w, graph.pos, graph.r
        N = r.shape[0]
        domain = graph.domain
        edge_index = graph.edge_index
        batch = graph.batch

        # use the current domain size and timestep and the increment as globals
        graph_features = torch.cat((graph.domain, graph.t, graph.x_global))
        graph_features_next = torch.cat((graph.domain_next, graph.t_next))
        graph_features = torch.cat((graph_features, graph_features_next))

        dynamic_features = torch.cat((pos, v, w), dim=1)
        features_without_position = torch.cat(
                (r, v, w, graph_features.repeat(N, 1),),
                dim=1)

        h = self.embedding_mlp(features_without_position)

        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, v, w, pos, r, graph_features, domain, edge_index, batch)

        model_output = self.output_mlp(h)
        model_output = torch.cat((
            torch.tensor([self.scale_position],device=self.device) * model_output[:, :3],
            torch.tensor([self.scale_velocity],device=self.device) * model_output[:, 3:-3],
            torch.tensor([self.scale_angular_velocity],device=self.device) * model_output[:, -3:],
            ), dim=1)

        prediction =  model_output + dynamic_features

        pred_x = torch.remainder(prediction[:, :3], graph.domain_next)
        pred_v = prediction[:, 3:-3]
        pred_w = prediction[:, -3:]

        h_mean = self.pool(h, batch)
        pred_macro = self.macro_mlp(h_mean)[0]  # gets rid of empty node dimension

        prediction = Prediction(
                positions=pred_x,
                velocities=pred_v,
                angular_velocities=pred_w,
                stress=pred_macro,
                )
        if detach:
            prediction = Prediction(*(el.detach() for el in prediction))
        return prediction


class NaiveForecasting(nn.Module):
    """
    Baseline prediction, predicting that the next step is the same as the current.
    (Don't predict the stress.)
    """
    def forward(self, graph: Graph) -> Prediction:
        return Prediction(
                positions=graph.pos,
                velocities=graph.v,
                angular_velocities=graph.w,
                stress=torch.zeros(3),
                )

