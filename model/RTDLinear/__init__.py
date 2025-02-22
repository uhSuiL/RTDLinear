from model.TDLinear.layer import TDLinear
from model.common import SimpleMovingAverage, Normalizer
from torch import nn


class Model(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.td_linear = TDLinear(
			moving_avg=SimpleMovingAverage(win_length=config.ma_win_len),
			time_embed_dim=config.time_embed_dim,
			num_step=config.num_step,
			num_pred_step=config.num_pred_step,
		)
		self.normalizer = Normalizer()

	def forward(self, model_inputs, cluster_id = None):
		if model_inputs.dim() == 2:
			model_inputs = model_inputs.unsqueeze(dim=0)

		# (batch_size, num_step, 1), (batch_size, num_step, num_series)
		time_ticks, time_series = model_inputs[..., 0].unsqueeze(dim=-1), model_inputs[..., 1].unsqueeze(dim=-1)

		normed_time_series = self.normalizer.transform(time_series)
		normed_pred_residual = self.td_linear(normed_time_series, time_ticks)
		normed_pred = normed_time_series[..., -1:, :] + normed_pred_residual

		pred = self.normalizer.restore(normed_pred)
		return pred.squeeze(dim=-1)
