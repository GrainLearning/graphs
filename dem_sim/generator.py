import torch
import torch_geometric as tg
from torch_cluster import radius_graph
from dem_sim.data_types import GraphData, Prediction, Graph


class GraphGenerator():
    def __init__(self, cutoff_distance: float, loop: bool = False) -> None:
        """
        Args:
            cutoff_distance (float): distance below which to connect two nodes.
            loop (bool): Whether to include self loops in the graph.
        """
        self.cutoff_distance = cutoff_distance
        self.loop = loop

    def build_graph(self, graph_data: GraphData, step: int = 0) -> Graph:
        """
        Create a single graph using time sequence data.

        Args:
            graph_data (GraphData): All tensor data of a given sample.
            step (int): which timestep to take.
        Returns:
            Graph
        """
        edge_index = self._compute_edges(
                graph_data.positions[step],
                graph_data.domain[step]
                )

        return Graph(
            pos=graph_data.positions[step],
            r=graph_data.radius,
            v=graph_data.velocities[step],
            w=graph_data.angular_velocities[step],
            t=graph_data.time[step],
            domain=graph_data.domain[step],
            t_next=graph_data.time[step + 1],
            domain_next=graph_data.domain[step + 1],
            x_global=graph_data.sample_properties,
            edge_index=edge_index,
        )

    def evolve(self,
            current_graph: Graph,
            prediction: Prediction,
            t_next: torch.tensor,
            domain_next: torch.tensor
            ) -> Graph:
        """
        Evolve graph to the next timestep using prediction and new inputs.

        Args:
            current_graph (Graph):
            prediction (Prediction):
            t_next (torch.tensor):
            domain_next (torch.tensor):

        Returns:
            Graph
        """
        edge_index = self._compute_edges(prediction.positions, domain_next)
        # todo: change to in-place when memory leak solved
        new_graph = Graph(
            pos=prediction.positions,
            r=current_graph.r,
            v=prediction.velocities,
            w=prediction.angular_velocities,
            t=current_graph.t_next,
            domain=current_graph.domain_next,
            t_next=t_next,
            domain_next=domain_next,
            x_global=current_graph.x_global,
            edge_index=edge_index,
            )
        return new_graph

    def _compute_edges(self, positions: torch.tensor, domain: torch.tensor) -> torch.tensor:
        """
        Compute the radius graph, taking into account periodic boundary conditions.

        This is done by applying `radius_graph` to the original positions and thrice more
        to the positions shifted and transformed back onto the domain.

        Args:
            positions (torch.tensor, shape [N, 3]): particle positions
            domain (torch.tensor, shape [3]): domain size.

        Returns:
            torch.tensor, shape [2, E]: edge list
        """
        edge_index = radius_graph(positions, self.cutoff_distance, loop=self.loop)

        for factor in [0.25, 0.5, 0.75]:
            shift_vector = factor * domain
            positions_shifted = torch.remainder(positions + shift_vector, domain)

            edge_index_shifted = radius_graph(positions_shifted, self.cutoff_distance, loop=self.loop)

            edge_index = torch.unique(torch.concat((edge_index, edge_index_shifted), dim=1), dim=1)

        return edge_index
