from typing import List
from collections import namedtuple
import torch
from torch_geometric.data import Data


GraphDataBase = namedtuple(
        "GraphData",
        ["sample_properties", "positions", "velocities", "angular_velocities", "radius", "time", "domain", "stress"],
        defaults=[None],
        )

PredictionBase = namedtuple(
        "Prediction",
        ["positions", "velocities", "angular_velocities", "stress"],
        )


class GraphData(GraphDataBase):
    """
    NamedTuple containing all the data of a sample in tensor form.

    Attributes:
        sample_properties ([4] tensor): constant graph-level features.
        positions ([N, 3] tensor)
        velocities ([N, 3] tensor)
        angular_velocities ([N, 3] tensor)
        radius ([N, 1] tensor): constant node features: particle radii.
        time ([T, 1] tensor): time deltas.
        domain ([T, 3] tensor): dynamical domain size.
        stress ([T, 3] tensor): graph level dynamical outputs.

    """
    __slots__ = ()
    def __repr__(self):
        repr_str = "GraphData("
        for field in self._fields:
            repr_str += f"{field}=tensor({getattr(self, field).shape}), "
        repr_str = repr_str[:-2] + ")"
        return repr_str


class Prediction(PredictionBase):
    """
    NamedTuple containing single step predictions.

    Attributes:
        positions ([N, 3] tensor)
        velocities ([N, 3] tensor)
        angular_velocities ([N, 3] tensor)
        stress ([3] tensor)
    """
    __slots__ = ()
    def __repr__(self):
        repr_str = "Prediction("
        for field in self._fields:
            repr_str += f"{field}=tensor({getattr(self, field).shape}), "
        repr_str = repr_str[:-2] + ")"
        return repr_str


class PredictionSequence():
    """
    Rollout prediction containing time series of predictions.

    Attributes:
        positions ([T, N, 3] tensor)
        velocities ([T, N, 3] tensor)
        angular_velocities ([T, N, 3] tensor)
        stress ([T, 3] tensor)
    """
    def __init__(self, prediction_list: List) -> None:
        self.positions = torch.stack(tuple(prediction.positions for prediction in prediction_list))
        self.velocities = torch.stack(tuple(prediction.velocities for prediction in prediction_list))
        self.angular_velocities = torch.stack(tuple(prediction.angular_velocities for prediction in prediction_list))
        self.stress = torch.stack(tuple(prediction.stress for prediction in prediction_list))

    def __repr__(self):
        repr_str = "Prediction("
        for field in PredictionBase._fields:
            repr_str += f"{field}=tensor({getattr(self, field).shape}), "
        repr_str = repr_str[:-2] + ")"
        return repr_str


class Graph(Data):
    """
    Wrapper around torch_geometric.data.Data.
    """
    def __init__(self, pos, r, v, w, t, domain, t_next, domain_next, x_global, edge_index):
        super().__init__(
                pos=pos,
                r=r,
                v=v,
                w=w,
                t=t,
                domain=domain,
                t_next=t_next,
                domain_next=domain_next,
                x_global=x_global,
                edge_index=edge_index,
                )

