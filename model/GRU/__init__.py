import torch

from torch import nn
from model.common import Normalizer, ClusterAwareAttention


class Model(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.gru = nn.GRU(
			input_size=config.input_dim + config.cluster_embed_dim if config.embed_cluster else config.input_dim,
			hidden_size=config.hidden_dim,
			num_layers=config.num_layer,
			batch_first=True,
		)
		self.normalizer = Normalizer()
		self.fc = nn.Linear(config.hidden_dim, config.output_dim)

		if config.embed_cluster:
			self.embed_cluster = nn.Embedding(
				num_embeddings=config.num_cluster,
				embedding_dim=config.cluster_embed_dim,
			)

	def forward(self, X: torch.tensor, cluster_id: int = None):
		if X.dim() == 2:
			X = X.unsqueeze(dim=0)  # (batch_size, num_step, input_dim)

		time_feat = X[..., 1:]
		X = X[..., 0].unsqueeze(dim=-1)

		X_norm = self.normalizer.transform(X)

		if self.config.embed_cluster:
			E_cluster = self.embed_cluster(cluster_id)
			feat = torch.concat([X_norm, time_feat, E_cluster], dim=-1)
		else:
			feat = torch.concat([X_norm, time_feat], dim=-1)
		output, hn = self.gru(feat)

		# hn = hn.unsqueeze(dim=1)  # (batch_size, 1, hidden_dim)
		hn = output[..., -1, :].unsqueeze(dim=-2)  # (batch_size, 1, hidden_dim)

		pred = self.fc(hn)
		pred = self.normalizer.restore(pred)
		return pred  # (batch_size, 1, output_dim)


class Model2(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.gru = nn.GRU(
			input_size=config.input_dim,
			hidden_size=config.hidden_dim,
			num_layers=config.num_layer,
			batch_first=True,
		)
		self.normalizer = Normalizer()
		self.fc = nn.Linear(config.hidden_dim, config.num_cluster if config.embed_cluster else 1)
		self.cluster_attn = ClusterAwareAttention(config.num_cluster, config.cluster_embed_dim)

	def forward(self, X: torch.tensor, cluster_id: torch.tensor = None):
		if X.dim() == 2:
			X = X.unsqueeze(dim=0)  # (batch_size, num_step, input_dim)
		if cluster_id is not None and cluster_id.dim() == 2:
			cluster_id = cluster_id[..., :1]  # (batch_size, 1)

		time_feat = X[..., 1:]
		X = X[..., 0].unsqueeze(dim=-1)

		X_norm = self.normalizer.transform(X)
		feat = torch.concat([X_norm, time_feat], dim=-1)

		output, hn = self.gru(feat)
		hn = output[..., -1, :].unsqueeze(dim=-2)  # (batch_size, 1, hidden_dim)

		preds = self.fc(hn)  # (batch_size, num_pred_step=1, num_cluster)
		preds = preds.transpose(-1, -2)
		pred = self.cluster_attn(preds, cluster_id)

		pred = self.normalizer.restore(pred)
		return pred  # (batch_size, 1, output_dim)
