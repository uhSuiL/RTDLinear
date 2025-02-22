import torch

from torch import nn, Tensor


class SMAPE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, pred: Tensor, true: Tensor) -> Tensor:
		assert pred.shape == true.shape, f'pred.shape={pred.shape}   true.shape={true.shape}'
		return torch.mean(2 * abs(pred - true) / (abs(pred) + abs(true)))


class EpochAwareLoss(nn.Module):
	def __init__(self, loss_fn: nn.Module, scale_fn = None):
		super().__init__()

		self.loss_func = loss_fn
		self.epoch = 0
		self.count_forward = 0
		self.scale_fn = scale_fn if scale_fn is not None else EpochAwareLoss.default_scale

	def forward(self, pred: Tensor, true: Tensor, epoch: int = None) -> Tensor:
		assert pred.shape == true.shape, f'pred.shape={pred.shape}   true.shape={true.shape}'
		self.count_forward += 1

		orig_loss = self.loss_func(pred, true)
		return orig_loss * self.scale_fn(epoch if epoch is not None else self.epoch)

	@staticmethod
	def default_scale(epoch: int):
		e150 = int(epoch / 150)
		if e150 > 0:
			return (1/10) ** (e150 * 0.9)
		return 1
