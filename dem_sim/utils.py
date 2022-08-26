import torch

def periodic_difference(x_i: torch.Tensor, x_j: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
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

def find_missing(A: torch.Tensor, B: torch.Tensor) -> List[tuple]:
    """
    Return edges from edge list A that are missing in edge list B.

    Args:
        A (torch.Tensor): edge list, shape [2, E].
        B (torch.Tensor): edge list, shape [2, E'].

    Returns:
        list of tuples
    """
    A_list = [tuple(edge.numpy()) for edge in list(A.T)]
    B_set = {tuple(edge.numpy()) for edge in list(B.T)}

    A_not_in_B = [edge for edge in A_list if edge not in B_set]
    return A_not_in_B

