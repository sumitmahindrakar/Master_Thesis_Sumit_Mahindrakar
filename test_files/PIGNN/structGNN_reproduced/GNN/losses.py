import torch
import torch.nn as nn

class L1_Loss(nn.Module):

    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, node_out, node_y, accuracy_threshold):

        condition = torch.abs(node_y) > accuracy_threshold

        loss = torch.abs(node_y[condition] - node_out[condition]).sum()

        return loss
    


class L2_Loss(nn.Module):

    def __init__(self):
        super(L2_Loss, self).__init__()

    def forward(self, node_out, node_y, accuracy_threshold):

        condition = torch.abs(node_y) > accuracy_threshold

        loss = (torch.abs(node_y[condition] - node_out[condition]) ** 2).sum()

        return loss