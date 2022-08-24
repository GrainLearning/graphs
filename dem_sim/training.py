import torch
from torch import nn
from torch.nn import MSELoss


def train_epoch(
        model,
        optimizer,
        dataset,
        loss_function,
        device: torch.device = 'cpu',
        ):
    optimizer.zero_grad()
    model.train()
    model.to(device)

    losses = []

    dataset.shuffle()
    for i, graph in enumerate(dataset):
        graph.to(device)
        pred_x, pred_v, pred_macro = model(graph)
        loss = loss_function(pred_macro, graph.y_global)
        loss.backward()
        print(f'Loss {loss}, step {i}...')
        print('pred_macro', pred_macro.detach())
        print('y', graph.y_global.detach())
        losses.append(loss.detach())
        optimizer.step()

    return losses


class DEMLoss(nn.Module):
    def __init__(self, a: float = 1., b: float = 1., c: float = 1.):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.loss_fn_x = MSELoss()
        self.loss_fn_v = MSELoss()
        self.loss_fn_macro = MSELoss()

    def forward(self, graph, preds_x, preds_v, preds_macro):
        next_x, next_v = graph.y[:, :3], graph.y[:, 3:]
        loss_x = self.loss_fn_x(preds_x, next_x)
        loss_v = self.loss_fn_v(preds_v, next_v)
        loss_macro = self.loss_fn_macro(preds_macro, graph.y_global)
        return self.a * loss_x + self.b * loss_v + self.c * loss_macro

