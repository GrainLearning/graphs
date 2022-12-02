import torch
from torch import nn
from torch.nn import MSELoss

from dem_sim.utils import periodic_difference
from dem_sim.data_types import Prediction, GraphData

from itertools import islice
import wandb

def train(
        simulator,
        optimizer,
        loader,
        loss_function,
        metric,
        device,
        epochs: int = 1,
        start_epoch: int = 0,
        start_step: int = 0,
        total_loss: float = 0.0
    ):
    optimizer.zero_grad()
    simulator.train()
    simulator.to(device)
    losses = []
    if start_step > 0: loader = islice(loader, start_step, None)

    for epoch in range(start_epoch, epochs):
        for i, (graph_data, step) in enumerate(loader, start = start_step):
            graph_data, step = _unbatch(graph_data, step)
            graph_data = graph_data.copy_to(device) #check if it is necessary after pin_memory=True in DataLoader definition

            prediction = simulator(graph_data, step)
            loss = loss_function(prediction, graph_data, step)
            metric.add(prediction, graph_data, step)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            if i>start_step and i%10 == 0: #Save model checkpoint
                wandb.log({"loss": loss}, step = i)
                torch.save({
                'epoch': epoch,
                'step': i,
                'model_state_dict': simulator.state_dict(), #check if this stores info recursevily (model).
                'optimizer_state_dict': optimizer.state_dict(),
                'metric_state_dict': metric.state_dict(),
                'total_loss_epoch': total_loss
                }, 'outputs/model.pth')
                print(f"Saved checkpoint at epoch {epoch}, step {i} to outputs/model.pth")
            
        total_loss = total_loss / (i + 1)
        losses.append(total_loss)

        print(f"Loss after epoch {epoch}: {total_loss:.4E}...")
        print(metric)

        wandb.log({"loss_epoch": total_loss, "epoch": epoch})
        wandb.log(metric.dict.update({"epoch" : epoch}))
        start_step = 0 # Reseting variables
        metric.reset()
        total_loss = 0

    return losses


def test(
        simulator,
        loader,
        loss_function,
        device,
    ):
    simulator.eval()
    simulator.to(device)
    metric = VectorMetrics()

    total_loss = 0
    for i, (graph_data, step) in enumerate(loader):
        graph_data, step = _unbatch(graph_data, step)
        prediction = simulator(graph_data, step)
        loss = loss_function(prediction, graph_data, step)
        metric.add(prediction, graph_data, step)
        total_loss += loss.item()
    total_loss = total_loss / (i + 1)
    print(f"Mean training loss: {total_loss:.4E}")
    print(metric)


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
    """
    Weighted sum of vector norm squared differences of positions, velocities, angular velocities and stress.
    """
    def __init__(self, a: float = 1., b: float = 1., c: float = 1., d: float = 1.):
        """
        Initialize a loss function.

        Args:
            a (float): multiplier of position loss.
            b (float): multiplier of velocity loss.
            c (float): multiplier of angular velocity loss.
            d (float): multiplier of stress loss.
        """
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.loss_fn_x = MSELossPeriodic()
        self.loss_fn_v = MSELoss()
        self.loss_fn_w = MSELoss()
        self.loss_fn_stress = MSELoss()

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
        loss_w = self.loss_fn_w(prediction.angular_velocities, graph_data.angular_velocities[step + 1])
        # Note we're predicting the stress at the current step, not the next one
        loss_stress = self.loss_fn_stress(prediction.stress, graph_data.stress[step])
        return 3 * (self.a * loss_x + self.b * loss_v + self.c * loss_w + self.d * loss_stress)


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


class VectorMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.positions = 0
        self.velocities = 0
        self.angular_velocities = 0
        self.stress = 0

    def reset(self):
        self.count = 0
        self.positions = 0
        self.velocities = 0
        self.angular_velocities = 0
        self.stress = 0

    def add(self, prediction: Prediction, graph_data: GraphData, step: int):
        self.positions += self._compute_mean_norm(prediction.positions, graph_data.positions[step + 1])
        self.velocities += self._compute_mean_norm(prediction.velocities, graph_data.velocities[step + 1])
        self.angular_velocities += self._compute_mean_norm(prediction.angular_velocities, graph_data.angular_velocities[step + 1])
        self.stress += self._compute_mean_norm(prediction.stress, graph_data.stress[step])
        self.count += 1

    def load_state_dict(self, state_dict: dict):
        for attribute in state_dict:
            setattr(self, attribute, state_dict[attribute])

    def __str__(self):
        output = "Mean vector norm differences: "
        for attribute in ['positions', 'velocities', 'angular_velocities', 'stress']:
            tabs = "".join(["\t" for _ in range((40 - len(attribute)) // 8)])
            mean = getattr(self, attribute) / self.count
            output += f"\n{attribute}:" + tabs + f"{mean:.4E}"
        return output

    def state_dict(self):
        state_dict = dict()
        for attribute in ['positions', 'velocities', 'angular_velocities', 'stress', 'count']:
            state_dict[attribute] = getattr(self,attribute)
        return state_dict

    @staticmethod
    def _compute_mean_norm(a, b):
        """
        Given two tensors a and b which contain (3 dimensional) vectors, the last dimension being the vector dimension,
        compute the norm of the difference vectors, and take the mean over the other dimensions.
        """
        diff_norms = ((a - b)**2).sum(dim=-1).sqrt().detach()
        return diff_norms.mean()

    @property
    def dict(self):
        """
        Returns the rms: root mean squared error of each of the attributes.
        """
        rms = dict()
        for attribute in ['positions', 'velocities', 'angular_velocities', 'stress']:
            mean = 0
            if self.count>0 :
                mean = getattr(self, attribute) / self.count
            rms['rms_'+attribute] = mean
        return rms
