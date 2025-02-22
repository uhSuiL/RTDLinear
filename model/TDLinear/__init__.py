from .layer import TDLinear
from model.common import SimpleMovingAverage, Normalizer, ClusterAwareAttention
from torch import nn

import torch
from time import time


class Model(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.td_linear = TDLinear(
			moving_avg=SimpleMovingAverage(win_length=config.ma_win_len),
			time_embed_dim=config.time_embed_dim,
			num_step=config.num_step,
			num_pred_step=config.num_pred_step,
			num_ex_t_var= 0 if not config.embed_cluster else config.cluster_embed_dim,
			num_ex_s_var= 0 if not config.embed_cluster else config.cluster_embed_dim,
		)
		self.normalizer = Normalizer()

		if config.embed_cluster:
			self.embed_cluster = nn.Embedding(config.num_cluster, config.cluster_embed_dim)

		self.fit_mid_preds = True

	def forward(self, model_inputs, cluster_id = None):
		if model_inputs.dim() == 2:
			model_inputs = model_inputs.unsqueeze(dim=0)
		if cluster_id is not None and cluster_id.dim() == 1:
			cluster_id = cluster_id.unsqueeze(dim=-1)  # (batch_size, num_series=1)

		# (batch_size, num_step, 1), (batch_size, num_step, num_series)
		with torch.no_grad():
			time_ticks, time_series = model_inputs[..., 0].unsqueeze(dim=-1), model_inputs[..., 1].unsqueeze(dim=-1)
			cluster_embed = None if not self.config.embed_cluster else self.embed_cluster(cluster_id)

			normed_time_series = self.normalizer.transform(time_series)
		normed_pred = self.td_linear(normed_time_series, time_ticks, Ex_t=cluster_embed, Ex_s=cluster_embed)
		pred = self.normalizer.restore(normed_pred)

		self.mid_season, self.mid_season_pred = self.td_linear.mid_season, self.td_linear.mid_season_pred

		return pred.squeeze(dim=-1)
