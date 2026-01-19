# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("l2_loss", dataclass=FairseqDataclass)
class GraphPredictionL1Loss(FairseqCriterion):
    """
    Implementation for the L2 loss (MAE loss) used in graphormer model training.
    """
    acc_loss, inc = 0, 0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        if isinstance(logits, tuple):
            logits, weights = logits
        else:
            weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)
        targets = model.get_targets(sample, [logits])
        # md data
        targets_normalize = (targets - 6.529300030461668) / 1.9919705951218716

        loss = nn.MSELoss(reduction="none")(logits, targets_normalize[: logits.size(0)])
        loss = (loss * weights).sum()

        y_true = targets_normalize[: logits.size(0)].reshape(-1)
        y_pred = logits.reshape(-1)
        weights_flat = weights.reshape(-1)
        err = y_pred - y_true
        sum_w = weights_flat.sum()
        sum_wy = (weights_flat * y_true).sum()
        sum_wy2 = (weights_flat * y_true * y_true).sum()
        sum_werr2 = (weights_flat * err * err).sum()

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
            "sum_w": sum_w,
            "sum_wy": sum_wy,
            "sum_wy2": sum_wy2,
            "sum_werr2": sum_werr2,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sum_w = sum(log.get("sum_w", 0) for log in logging_outputs)
        sum_wy = sum(log.get("sum_wy", 0) for log in logging_outputs)
        sum_wy2 = sum(log.get("sum_wy2", 0) for log in logging_outputs)
        sum_werr2 = sum(log.get("sum_werr2", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        sum_w_value = float(sum_w) if torch.is_tensor(sum_w) else sum_w
        if sum_w_value > 0:
            mean_y = sum_wy / sum_w
            sst = sum_wy2 - sum_w * mean_y * mean_y
            sst_value = float(sst) if torch.is_tensor(sst) else sst
            if sst_value > 0:
                r2 = 1.0 - (sum_werr2 / sst)
            else:
                r2 = 0.0
            metrics.log_scalar("r2", r2, sum_w, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("l2_loss_with_flag", dataclass=FairseqDataclass)
class GraphPredictionL1LossWithFlag(GraphPredictionL1Loss):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)
        if isinstance(logits, tuple):
            logits, weights = logits
        else:
            weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)
        targets = model.get_targets(sample, [logits])
        # md data
        targets_normalize = (targets - 6.529300030461668) / 1.9919705951218716

        loss = nn.MSELoss(reduction="none")(logits, targets_normalize[: logits.size(0)])
        loss = (loss * weights).sum()

        y_true = targets_normalize[: logits.size(0)].reshape(-1)
        y_pred = logits.reshape(-1)
        weights_flat = weights.reshape(-1)
        err = y_pred - y_true
        sum_w = weights_flat.sum()
        sum_wy = (weights_flat * y_true).sum()
        sum_wy2 = (weights_flat * y_true * y_true).sum()
        sum_werr2 = (weights_flat * err * err).sum()

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
            "sum_w": sum_w,
            "sum_wy": sum_wy,
            "sum_wy2": sum_wy2,
            "sum_werr2": sum_werr2,
        }
        return loss, sample_size, logging_output
