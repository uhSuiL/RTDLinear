import torch

from model.common import Normalizer, SimpleMovingAverage, ClusterAwareAttention
from .layer import DLinear

from torch import nn


class Model(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.num_pred_step = config.num_pred_step
		self.keep_num_pred_dim = False

		self.normalizer = Normalizer()
		self.d_linear = DLinear(
			True, 1, config.num_step, config.num_pred_step,
			SimpleMovingAverage(win_length=config.ma_win_len),
			num_ex_t_var=0 if not config.embed_cluster else config.cluster_embed_dim,
			num_ex_s_var=0 if not config.embed_cluster else config.cluster_embed_dim,
		)

		if config.embed_cluster:
			self.embed_cluster = nn.Embedding(config.num_cluster, config.cluster_embed_dim)

	def forward(self, time_series: torch.Tensor, cluster_id = None):
		normed_time_series = self.normalizer.transform(time_series)

		if self.config.embed_cluster:
			cluster_embedding = self.embed_cluster(cluster_id)  # (batch_size, embed_dim)
			if cluster_embedding.dim() == 2:
				cluster_embedding = cluster_embedding.unsqueeze(dim=-2)  # (batch_size, num_series=1, embed_dim)
			pred = self.d_linear(normed_time_series, cluster_embedding, cluster_embedding)
		else:
			pred = self.d_linear(normed_time_series)

		pred = self.normalizer.restore(pred)
		return pred.squeeze(dim=1) if (self.num_pred_step == 1) and (not self.keep_num_pred_dim) else pred


class Model2(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.num_cluster = config.num_cluster
		self.num_pred_step = config.num_pred_step
		self.keep_num_pred_dim = False

		self.normalizer = Normalizer()
		self.d_linear = DLinear(
			True,
			1 if not config.embed_cluster else config.num_cluster,
			config.num_step,
			config.num_pred_step,
			SimpleMovingAverage(win_length=config.ma_win_len),
		)
		self.cluster_attn = ClusterAwareAttention(num_cluster=config.num_cluster, embed_dim=config.cluster_embed_dim)

	def forward(self, time_series: torch.Tensor, cluster_id = None):
		"""
		:param time_series: (batch_size, num_step, num_series=1)
		:param cluster_id: (batch_size, )
		:return: pred: (batch_size, [num_pred_step])
		"""
		assert cluster_id.dim() == 1, cluster_id.dim()
		assert time_series.shape[-1] == 1, time_series.shape

		normed_time_series = self.normalizer.transform(time_series)

		if self.config.embed_cluster:
			assert cluster_id is not None
			normed_time_series = normed_time_series.repeat_interleave(self.num_cluster, dim=-1)  # (batch_size, num_step, num_cluster)
			preds = self.d_linear(normed_time_series)  # (batch_size, num_pred_step, num_cluster)
			preds = preds.transpose(-1, -2)  # (batch_size, num_cluster, num_pred_step)
			pred = self.cluster_attn(preds, cluster_id.unsqueeze(dim=-1))  # (batch_size, num_pred_step)
		else:
			pred = self.d_linear(normed_time_series)

		assert pred.dim() == time_series.dim(), """WARNING: dim mismatch for normalizer restore"""
		pred = self.normalizer.restore(pred)
		return pred.squeeze(dim=1) if (self.num_pred_step == 1) and (not self.keep_num_pred_dim) else pred
