import torch
from torch import nn

from dltranz.seq_encoder import PaddedBatch


class PairwiseMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        """
        Pairwise Margin Ranking Loss. All setted parameters redirected to nn.MarginRankingLoss.
        All the difference is that pairs automatically generated for margin ranking loss.
        All possible pairs of different class are generated.
        """
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin, size_average, reduce, reduction)

    def forward(self, prediction, label):
        """
        Get pairwise margin ranking loss.
        :param prediction: tensor of shape Bx1 of predicted probabilities
        :param label: tensor of shape Bx1 of true labels for pair generation
        """

        # positive-negative selectors
        mask_0 = label == 0
        mask_1 = label == 1

        # selected predictions
        pred_0 = torch.masked_select(prediction, mask_0)
        pred_1 = torch.masked_select(prediction, mask_1)
        pred_1_n = pred_1.size()[0]
        pred_0_n = pred_0.size()[0]

        if pred_1_n > 0 and pred_0_n:
            # create pairs
            pred_00 = pred_0.unsqueeze(0).repeat(1, pred_1_n)
            pred_11 = pred_1.unsqueeze(1).repeat(1, pred_0_n).view(pred_00.size())
            out01 = -1 * torch.ones(pred_1_n*pred_0_n).to(prediction.device)

            return self.margin_loss(pred_00.view(-1), pred_11.view(-1), out01)
        else:
            return torch.sum(prediction) * 0.0


class MultiLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, pred, true):
        loss = 0
        for weight, criterion in self.losses:
            loss = weight * criterion(pred, true) + loss

        return loss


class AllStateLoss(nn.Module):
    def __init__(self, point_loss):
        super().__init__()
        self.loss = point_loss

    def forward(self, pred: PaddedBatch, true):
        y = torch.cat([torch.Tensor([yb] * length) for yb, length in zip(true, pred.seq_lens)])
        weights = torch.cat([torch.arange(1, length + 1) / length for length in pred.seq_lens])

        loss = self.loss(pred, y, weights)

        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


def get_loss(params):
    loss_type = params['train.loss']

    if loss_type == 'bce':
        loss = BCELoss()
    elif loss_type == 'ranking':
        loss = PairwiseMarginRankingLoss(margin=params['ranking.loss_margin'])
    elif loss_type == 'both':
        loss = MultiLoss([(1, BCELoss()), (1, PairwiseMarginRankingLoss(margin=params['ranking.loss_margin']))])
    elif loss_type == 'mae':
        loss = nn.L1Loss()
    elif loss_type == 'mse':
        loss = nn.MSELoss()
    else:
        raise Exception(f'unknown loss type: {loss_type}')

    if params.get('head',{}).get('pred_all_states'):
        loss = AllStateLoss(loss)

    return loss
