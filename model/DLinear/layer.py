import torch
from torch import nn


class DLinear(nn.Module):
	def __init__(self,
				 is_individual: bool, num_series: int, num_step: int, num_pred_step: int,
				 moving_avg: nn.Module = None,
				 num_ex_t_var: int = 0, num_ex_s_var: int = 0):
		super().__init__()
		self.is_individual = is_individual

		init_w_shape = (num_series, num_pred_step, num_step) if is_individual else (num_pred_step, num_step)
		init_weights = torch.ones(*init_w_shape) / num_step

		init_w_et_shape = (num_series, num_pred_step, num_ex_t_var) if is_individual \
			else (num_pred_step, num_step + num_ex_t_var)
		init_et_weight = torch.ones(*init_w_et_shape)

		init_w_es_shape = (num_series, num_pred_step, num_ex_s_var) if is_individual \
			else (num_pred_step, num_ex_s_var)
		init_es_weight = torch.ones(*init_w_es_shape)

		self.W_t = nn.Parameter(torch.concat([init_weights, init_et_weight], dim=-1))
		self.b_t = nn.Parameter(torch.zeros(num_pred_step, num_series))

		self.W_s = nn.Parameter(torch.concat([init_weights, init_es_weight], dim=-1))
		self.b_s = nn.Parameter(torch.zeros(num_pred_step, num_series))

		self.moving_avg = moving_avg

	def reshape_ex_var(self, Ex_var: torch.Tensor):
		if Ex_var.dim() == 2:
			Ex_var = Ex_var.unsqueeze(dim=0)  # (batch_size, num_series, num_vars)
		return Ex_var.unsqueeze(dim=-1)  # (batch_size, num_series, num_vars, 1)

	def forward(self, X: torch.Tensor, Ex_t: torch.Tensor = None, Ex_s: torch.Tensor = None):
		"""ATTENTION: MAKE SURE DIMENSION INCLUDES `num_series`
			Shape
				- X: ([batch_size], num_step, num_series)
				- exo_vars: ([batch_size], num_series, num_vars)
		"""
		if X.dim() == 2:  # set `batch_size = 1`
			X = X.unsqueeze(dim=0)  # (1, num_step, num_series)

		X_t = self.moving_avg(X)  # (batch_size, num_steps, num_series)
		X_s = X - X_t  # (batch_size, num_step, num_series)

		X_t = X_t.permute(0, 2, 1).unsqueeze(dim=-1)  # (batch_size, num_series, num_step, 1)
		if Ex_t is not None:
			Ex_t = self.reshape_ex_var(Ex_t)  # (batch_size, num_series, num_vars, 1)
			X_t = torch.concat([X_t, Ex_t], dim=-2)
			# (batch_size, num_series, num_steps + num_vars, 1)

		X_s = X_s.permute(0, 2, 1).unsqueeze(dim=-1)  # (batch_size, num_series, num_step, 1)
		if Ex_s is not None:
			Ex_s = self.reshape_ex_var(Ex_s)  # (batch_size, num_series, num_vars, 1)
			X_s = torch.concat([X_s, Ex_s], dim=-2)
			# (batch_size, num_series, num_step + num_vars, 1)

		# (num_series, num_pred_step, num_step+[num_ex_s_vars]) x (batch_size, num_series, num_step+[num_ex_s_vars], 1)
		H_t = self.W_t @ X_t  # (batch_size, num_series, num_pred_step, 1)
		H_s = self.W_s @ X_s  # (batch_size, num_series, num_pred_step, 1)

		H_t = H_t.squeeze(dim=-1).permute(0, 2, 1)  # (batch_size, num_pred_step, num_series)
		H_s = H_s.squeeze(dim=-1).permute(0, 2, 1)  # (batch_size, num_pred_step, num_series)

		X_hat = (H_t + self.b_t) + (H_s + self.b_s)  # (batch_size, num_pred_step, num_series)
		return X_hat
