import torch
import numpy as np

from math import sqrt, pi

from torch import nn
from torch.nn import functional as F


class Normalizer:
	"""
	Default:
		Normalize each feature data among num_step
	"""
	def __init__(self, dim = -2):
		self.mean = 0
		self.std = 1
		self.dim = dim

	def transform(self, data: torch.tensor, dim: int = -2, *, is_baseline=True) -> torch.tensor:
		"""
		Default:
			data: shape=(batch_size, num_steps, feature_dim)
			dim: 1
			Normalize each feature data among num_step
		"""
		self.dim = dim
		if is_baseline:
			self.mean = torch.mean(data, dim=self.dim, keepdim=True)
			self.std = torch.std(data, dim=self.dim, keepdim=True)
		return (data - self.mean) / self.std

	def restore(self, data: torch.tensor) -> torch.tensor:
		return data * self.std + self.mean


class SimpleMovingAverage(nn.Module):
	def __init__(self, win_length: int, padding=None):
		super().__init__()
		self.padding = padding
		self.win_length = win_length
		self.moving_avg = nn.AvgPool1d(kernel_size=win_length, stride=1)

	def forward(self, X: torch.Tensor):  # (batch_size, num_steps, num_features)
		if X.dim() == 2:  # (num_steps, num_features)
			X = torch.unsqueeze(X, dim=0)  # (1, num_steps, num_features)

		X = X.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		ma = self.moving_avg(X)  # (batch_size, num_features, num_steps - (win_length - 1))
		# Keep sequence length
		if self.padding is None:
			X = torch.concat([
				X[:, :, : self.win_length - 1],
				ma
			], dim=-1)  # (batch_size, num_features, num_steps)
		elif self.padding == 'avg':
			X = torch.concat([
				ma[:, :, :1].repeat_interleave(self.win_length - 1, -1),
				ma
			], dim=-1)

		X = X.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		return X


class WeightedMovingAverage(nn.Module):
	def __init__(self, win_length: int, num_features: int):
		super().__init__()
		self.win_length = win_length
		self.moving_avg = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=win_length)

	def forward(self, X: torch.Tensor):  # (batch_size, num_steps, num_features)
		if X.dim() == 2:  # (num_steps, num_features)
			X = torch.unsqueeze(X, dim=0)  # (1, num_steps, num_features)

		X = X.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		X_smooth = torch.concat([
			X[:, :, :self.win_length - 1],
			self.moving_avg(X)
		], dim=-1)  # (batch_size, num_features, num_steps)

		X_smooth = X_smooth.permute(0, 2, 1)  # (batch_size, num_features, num_steps)
		return X_smooth


class UltimateSmoother(nn.Module):
	"""refer: https://www.mesasoftware.com/papers/UltimateSmoother.pdf"""
	def __init__(self, period: int | list = 20):
		super().__init__()
		self.period = torch.tensor(period, dtype=torch.float)  # (1,) | (num_features,)

	def forward(self, X):  # (batch_size, num_steps, num_features)
		a1 = torch.exp(-sqrt(2) * pi / self.period)
		c3 = - a1 ** 2
		c2 = -2 * a1 * torch.cos(sqrt(2) * 180 / self.period)
		c1 = (1 + c2 - c3) / 4

		if X.dim() == 2:  # (num_steps, num_features)
			X = torch.unsqueeze(X, dim=0)  # (1, num_steps, num_features)

		X = X.permute(1, 0, 2)  # (num_steps, batch_size, num_features)

		X_smooth = X.clone()
		for t in range(X.shape[0])[3:]:
			X_smooth[t] = (
					(1 - c1) * X[t - 1]
					+ (2 * c1 - c2) * X[t - 2]
					- (c1 + c3) * X[t - 3]
					+ c2 * X_smooth[t - 1]
					+ c3 * X_smooth[t - 2]
			)

		X_smooth = X_smooth.permute(1, 0, 2)  # (batch_size, num_steps, num_features)
		return X_smooth

	@staticmethod
	def ultimate_smooth(X: np.ndarray, period: int = 20, auto_fill: bool = False):
		"""numpy version for direct smooth operation"""
		# X shape: (T, )
		a1 = np.exp(-sqrt(2) * pi / period)
		c3 = -a1 ** 2
		c2 = -2 * a1 * np.cos(sqrt(2) * 180 / period)
		c1 = (1 + c2 - c3) / 4

		X_smooth = X.copy()
		for t in range(X.shape[0])[3:]:
			X_smooth[t] = (
				(1 - c1) * X[t - 1]
				+ (2 * c1 - c2) * X[t - 2]
				- (c1 + c3) * X[t - 3]
				+ c2 * X_smooth[t - 1]
				+ c3 * X_smooth[t - 2]
			)

		return X_smooth if auto_fill else X_smooth[3:]


def MLP(layer_dims: list, Activation: nn.Module = None):
	assert len(layer_dims) > 1, len(layer_dims)

	Activation = nn.LeakyReLU if Activation is None else Activation
	layers = []
	for i in range(len(layer_dims) - 2):
		layers += [nn.Linear(layer_dims[i], layer_dims[i + 1]), Activation()]
	layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
	return nn.Sequential(*layers)


class Time2Vec(nn.Module):
	"""(batch_size, num_steps, 1) -> (batch_size, num_steps, num_series, embed_dim)

	Different from https://arxiv.org/abs/1907.05321, our implementation further consider:
	  - For Multi-variate TSF, each series may have its own time embedding -> num_series
	  - For time-independent (maybe series-dependent) variables, embed and add them to `time_embed`
	"""
	def __init__(
			self,
			embed_dim: int,
			num_series: int = 1,
			activation_fn=None,
			keep_dim_series: bool = False,
			num_ex_var: int = 0):
		super().__init__()

		self.f = torch.sin if activation_fn is None else activation_fn
		# self.omg = nn.Parameter(torch.randn(1, embed_dim * num_series))
		# self.phi = nn.Parameter(torch.randn(num_series, embed_dim))

		self.omg = nn.Parameter(torch.ones(1, embed_dim * num_series) * 0.618)
		self.phi = nn.Parameter(torch.zeros(num_series, embed_dim))

		if num_ex_var > 0:
			self.ex_embedding = nn.Linear(num_ex_var, embed_dim, bias=False)

		self.num_series = num_series
		self.keep_dim_series = keep_dim_series
		self.num_ex_var = num_ex_var

	def forward(self, time_ticks: torch.Tensor, ex_var: torch.tensor = None):
		"""
		:param time_ticks: ([batch_size], num_steps, 1)
		:param ex_var: ([batch_size], num_series, num_var)
		"""
		assert time_ticks.shape[-1] == 1, """ATTENTION: KEEP THE LAST DIM `1`"""
		if time_ticks.dim() == 2:
			time_ticks = torch.unsqueeze(time_ticks, dim=0)

		time_embed = time_ticks @ self.omg  # (batch_size, num_steps, embed_dim*num_series)
		time_embed = time_embed.reshape(
			*time_embed.shape[:2], self.num_series, -1)  # (batch_size, num_steps, num_series, embed_dim)
		time_embed += self.phi

		time_embed = torch.concat([
			time_embed[..., :1],
			self.f(time_embed[..., 1:]),
		], dim=-1)  # (batch_size, num_steps, num_series, embed_dim)

		if ex_var is not None and self.num_ex_var > 0:
			ex_embed = self.ex_embedding(ex_var)  # (batch_size, num_series, embed_dim)
			ex_embed = ex_embed \
				.unsqueeze(dim=-3) \
				.repeat_interleave(time_embed.shape[-3], dim=-3)  # (batch_size, num_step, num_series, embed_dim)
			time_embed += ex_embed

		return time_embed if self.keep_dim_series else time_embed.squeeze(dim=-2)
		# ([batch_size], num_steps, [num_series], embed_dim)


class ClusterAwareAttention(nn.Module):
	def __init__(self, num_cluster, embed_dim, keep_dim = True):
		super().__init__()
		self.keep_dim = keep_dim

		self.cluster_embedding = nn.Embedding(num_embeddings=num_cluster, embedding_dim=embed_dim)
		self.all_cluster_idx = torch.arange(num_cluster).reshape(1, -1)

	def forward(self, preds: torch.tensor, cluster_idx: torch.tensor):
		"""
		:param preds: (batch_size, num_cluster, num_pred_step)
		:param cluster_idx: (batch_size, 1)
		:return: pred: (batch_size, [1], num_pred_step)
		"""
		assert preds.dim() == 3, """Illegal Input: preds dim should be 3"""
		assert cluster_idx.shape[-1] == 1, """ATTENTION: keep `cluster_id` last dim = 1"""

		all_cluster_idx = self.all_cluster_idx.repeat_interleave(cluster_idx.shape[0], dim=0).to(cluster_idx.device)
		all_cluster_embedding = self.cluster_embedding(all_cluster_idx)  # (batch_size, num_cluster, embed_dim)
		cluster_embedding = self.cluster_embedding(cluster_idx)  # (batch_size, 1, embed_dim)

		pred = F.scaled_dot_product_attention(
			query=cluster_embedding,  # (batch_size, 1, embed_dim)
			key=all_cluster_embedding,  # (batch_size, num_cluster, embed_dim)
			value=preds,  # (batch_size, num_cluster, num_pred_step)
		)  # (batch_size, 1, num_pred_step)
		assert pred.shape[1] == cluster_embedding.shape[1] and pred.shape[2] == preds.shape[2]

		return pred if self.keep_dim else pred.squeeze(dim=-2)
