import torch
from dem_sim.data_types import GraphData, Prediction, PredictionSequence


class Simulator(torch.nn.Module):
    def __init__(self, model, graph_generator) -> None:
        super().__init__()
        self.model = model
        self.graph_generator = graph_generator

    def forward(self, step: int, graph_data: GraphData) -> Prediction:
        graph = self.graph_generator.build_graph(graph_data, step)
        return self.model(graph)

    def rollout(self,
            initial_data: GraphData,
            domain_sequence: torch.Tensor,
            time_sequence: torch.Tensor,
            ) -> Prediction:
        graph = self.graph_generator.build_graph(initial_data)

        T = time_sequence.shape[0]
        predictions = []

        for t in range(T - 1):  # TODO: think about range ends
            print(t)
            prediction = self.model(graph)
            predictions.append(prediction)
            graph = self.graph_generator.evolve(graph, prediction, time_sequence[t + 1], domain_sequence[t + 1])

        return PredictionSequence(predictions)

