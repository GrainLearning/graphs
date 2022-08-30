import torch
from torch import nn

from dem_sim.layers import GNN_Layer, GlobalMeanPool
from dem_sim.data_types import Prediction, Graph


class GNNModel(nn.Module):
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

        self.pool = GlobalMeanPool()
        self.macro_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.output_macro_features),
        )

    def forward(self, graph: Graph) -> Prediction:
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

        h_mean = self.pool(h, batch)
        pred_macro = self.macro_mlp(h_mean)[0]  # gets rid of empty node dimension

        return Prediction(positions=pred_x, velocities=pred_v, stress=pred_macro)


class NaiveForecasting(nn.Module):
    """
    Baseline prediction, predicting that the next step is the same as the current.
    (Don't predict the stress.)
    """
    def forward(self, graph: Graph) -> Prediction:
        return Prediction(
                positions=graph.pos,
                velocities=graph.v,
                stress=torch.zeros(3),
                )

