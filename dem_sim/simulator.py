import torch
from dem_sim.data_types import GraphData, Prediction, PredictionSequence


class Simulator(torch.nn.Module):
    def __init__(self, model, graph_generator) -> None:
        super().__init__()
        self.model = model
        self.graph_generator = graph_generator

    def forward(self, graph_data: GraphData, step: int) -> Prediction:
        graph = self.graph_generator.build_graph(graph_data, step)
        return self.model(graph)

    def rollout(self,
            initial_data: GraphData,
            domain_sequence: torch.tensor,
            time_sequence: torch.tensor,
            ) -> Prediction:
        graph = self.graph_generator.build_graph(initial_data, 0)

        T = time_sequence.shape[0]
        predictions = []

        for t in range(T - 1):
            print(t)
            prediction = self.model(graph, detach=True)
            predictions.append(prediction)
            graph = self.graph_generator.evolve(graph, prediction, time_sequence[t + 1], domain_sequence[t + 1])

        return PredictionSequence(predictions)

