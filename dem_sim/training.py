import torch
from torch import nn
from torch.nn import MSELoss

from dem_sim.utils import periodic_difference
from dem_sim.data_types import Prediction, GraphData


def train(
        simulator,
        optimizer,
        loader,
        loss_function,
        device,
        epochs: int = 1,
    ):
    optimizer.zero_grad()
    simulator.train()
    simulator.to(device)

    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for i, (graph_data, step) in enumerate(loader):
            graph_data, step = _unbatch(graph_data, step)
            prediction = simulator(graph_data, step)
            loss = loss_function(prediction, graph_data, step)
            loss.backward()
            #print(f'Loss {loss}, step {i}...')
            #print('pred_macro', prediction.stress.item())
            #print('y', graph_data.stress[step].item())
            total_loss += loss.item()
            optimizer.step()
        total_loss = total_loss / (i + 1)
        losses.append(total_loss)
        print(f"Loss after epoch {epoch}: {total_loss}...")

    return losses


def test(
        simulator,
        loader,
        loss_function,
        device,
    ):
    simulator.eval()
    simulator.to(device)

    total_loss = 0
    for i, (graph_data, step) in enumerate(loader):
        graph_data, step = _unbatch(graph_data, step)
        prediction = simulator(graph_data, step)
        loss = loss_function(prediction, graph_data, step)
        total_loss += loss.item()
    total_loss = total_loss / (i + 1)
    print(f"Mean training loss: {total_loss}")


def _unbatch(graph_data, step):
    """
    Temporary code to remove the batch dimension added by the dataloader.
    To be replaced with actual handling of batches.
    """
    if type(step) != int:
        step = step[0]
        graph_data = GraphData(**{key: getattr(graph_data, key)[0] for key in GraphData._fields})
    return graph_data, step

class DEMLoss(nn.Module):
    def __init__(self, a: float = 1., b: float = 1., c: float = 1.):
        """
        Initialize a loss function.

        Args:
            a (float): multiplier of position loss.
            b (float): multiplier of velocity loss.
            c (float): multiplier of macroscopic loss.
        """
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.loss_fn_x = MSELossPeriodic()
        self.loss_fn_v = MSELoss()
        self.loss_fn_macro = MSELoss()

    def forward(self, prediction: Prediction, graph_data: GraphData, step: int):
        """
        Compute the loss.

        Args:
            prediction (Prediction): The model predictions.
            graph_data (GraphData): The tensor data of the grpah, time sequences.
            step (int): The step that the prediction was made on.

        Returns:
            float loss
        """
        loss_x = self.loss_fn_x(prediction.positions, graph_data.positions[step + 1], graph_data.domain[step + 1])
        loss_v = self.loss_fn_v(prediction.velocities, graph_data.velocities[step + 1])
        # Note we're predicting the stress at the current step, not the next one
        loss_macro = self.loss_fn_macro(prediction.stress, graph_data.stress[step])
        return self.a * loss_x + self.b * loss_v + self.c * loss_macro


class MSELossPeriodic(MSELoss):
    """
    Mean squared error loss, but taking into account periodicity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, domain: torch.Tensor):
        error = periodic_difference(inputs, target, domain)
        se = error ** 2
        mse = torch.mean(se)
        return mse

