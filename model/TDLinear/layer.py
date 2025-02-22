import torch
from time import time

from model.common import Time2Vec
from torch import nn
from torch.nn import functional as F


class TDLinear(nn.Module):
	def __init__(self,
				 moving_avg: nn.Module,
				 time_embed_dim,
				 num_step,
				 num_pred_step,
				 num_series = 1,
				 num_ex_t_var = 0,
				 num_ex_s_var = 0,
				 is_individual = True):
		super().__init__()
		num_step -= moving_avg.win_length  # TODO: TEST TDLINEAR
		init_w_shape = (num_series, num_pred_step, num_step + num_ex_t_var) if is_individual \
			else (num_pred_step + num_ex_t_var, num_step)
		init_weights = torch.ones(*init_w_shape) / num_step
		self.W_t = nn.Parameter(init_weights)
		self.b_t = nn.Parameter(torch.zeros(num_pred_step, num_series))

		self.moving_avg = moving_avg
		self.t2v = Time2Vec(time_embed_dim, num_ex_var=num_ex_s_var)

		self.num_pred_step = num_pred_step
		self.num_ex_t_var = num_ex_t_var

	def forward(self, X: torch.tensor, T: torch.tensor, Ex_t = None, Ex_s = None):
		"""
		:param X: (batch_size, num_step, num_series)
		:param T: (batch_size, num_step, 1)
		:param Ex_t: (batch_size, num_series, num_var)
		:param Ex_s: (batch_size, num_series, num_var)
		"""
		with torch.no_grad():
			T_next = T[..., -1:, :] + torch.arange(1, self.num_pred_step + 1).to(T.device)  # (batch_size, num_pred_step, 1)

		X_trend = self.moving_avg(X)  # ([batch_size], num_step, num_series)
		X_season = X - X_trend  # ([batch_size], num_step, num_series)

		# TODO: TEST TDLINEAR
		X_trend, X_season = X_trend[:, self.moving_avg.win_length:, :], X_season[:, self.moving_avg.win_length:, :]
		T = T[:, self.moving_avg.win_length:, :]

		X_trend = X_trend.transpose(-1, -2).unsqueeze(dim=-1)  # (batch_size, num_series, num_step, 1)

		if Ex_t is not None and self.num_ex_t_var > 0:
			Ex_t = Ex_t.unsqueeze(dim=-1)  # (batch_size, num_series, num_var, 1)
			X_trend = torch.concat([X_trend, Ex_t], dim=-2)	 # (batch_size, num_series, num_step+num_var, 1)

		pred_trend = self.W_t @ X_trend  # (batch_size, num_series, num_pred_step, 1)
		pred_trend = pred_trend.squeeze(dim=-1).transpose(-1, -2)  # (batch_size, num_pred_step, num_series)
		pred_trend += self.b_t  # (batch_size, num_pred_step, num_series)

		embed_t = self.t2v(T, Ex_s)  # (batch_size, num_step, embed_dim)
		embed_t_next = self.t2v(T_next, Ex_s)  # (batch_size, num_pred_step, embed_dim)

		pred_season = F.scaled_dot_product_attention(
			query=embed_t_next,  # (batch_size, num_pred_step, embed_dim)
			key=embed_t,  # (batch_size, num_step, embed_dim)
			value=X_season  # (batch_size, num_step, num_series)
		)  # (batch_size, num_pred_step, num_series)

		mid_range = 7
		self.mid_season = X_season[:, -mid_range:, :]  # (batch_size, mid_range, num_series=1)
		self.mid_season_pred = torch.concat([
			F.scaled_dot_product_attention(query=embed_t[:, t:t+1, :], key=embed_t[:, :t, :], value=X_season[:, :t, :])
			for t in range(embed_t.shape[1])[-mid_range:]
		], dim=1)  # (batch_size, mid_range, num_series=1)

		# # Track mid output
		# torch.save({
		# 	'X': X, 'T': T,
		# 	'X_trend': X_trend,
		# 	'X_season': X_season,
		# 	'pred_trend': pred_trend,
		# 	'pred_season': pred_season,
		# 	'embed_t': embed_t,
		# 	'embed_t_next': embed_t_next,
		# }, f'TDLinear_layer_{time()}.pt')

		pred = pred_trend + pred_season  # (batch_size, num_pred_step, num_series)
		return pred  # (batch_size, num_pred_step, num_series)
