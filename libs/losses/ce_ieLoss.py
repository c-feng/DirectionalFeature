import torch
import torch.nn as nn

class CE_IELoss(nn.Module):
    """
    CrossEntropy Loss with Information Entropy as a regularization
    """
    def __init__(self, eps=0.5, reduction='mean'):
        super(CE_IELoss, self).__init__()
        self.eps = eps
        self.nll = nn.NLLLoss(reduction=reduction)
        self.softmax = nn.Softmax(1)

    def update_eps(self):
        self.eps = self.eps * 0.1

    def forward(self, outputs, labels):
        """
        :param outputs: [b, c]
        :param labels: [b,]
        :return: a loss (Variable)
        """
        outputs = self.softmax(outputs)  # probabilities
        ce = self.nll(outputs.log(), labels)
        reg = outputs * outputs.log()
        reg = reg.sum(1).mean()
        loss_total = ce + reg * self.eps
        return loss_total #, ce, reg