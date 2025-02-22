import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.util import DecompSeries
from util import *


class SlidingWinDataset(Dataset):
	def __init__(self, time_series: pd.DataFrame, sliding_win: int):
		time_series = time_series.to_numpy(dtype=np.float32)

		# sliding_win -> time_series length
		# predict s[t] based on s[t-1]~s[t-1-win]
		self.windows = [
			torch.tensor(time_series[t - sliding_win: t])
			for t in range(sliding_win, time_series.shape[0])
		]
		self.labels = [
			torch.tensor(time_series[t])
			for t in range(sliding_win, time_series.shape[0])
		]

	def __getitem__(self, win_id):
		return self.labels[win_id], self.windows[win_id]

	def __len__(self):
		assert len(self.windows) == len(self.labels), f'{len(self.windows), len(self.labels)}'
		return len(self.windows)


class ClusterDataset(Dataset):
	"""gather a cluster of dataset into one dataset"""
	def __init__(self, dataset_cluster: list[Dataset], return_ds_id: bool = True):
		self.dataset_cluster = dataset_cluster
		self.return_ds_id = return_ds_id
		self.ds_len_list = [len(ds) for ds in dataset_cluster]

	def __getitem__(self, idx):
		cum_len = 0
		for ds_id, ds_len in enumerate(self.ds_len_list):
			if cum_len <= idx < cum_len + ds_len:
				if self.return_ds_id:
					return ds_id, self.dataset_cluster[ds_id][idx - cum_len]
				else:
					return self.dataset_cluster[ds_id][idx - cum_len]

			cum_len += ds_len

	def __len__(self):
		return sum(self.ds_len_list)


class DecompSlidingWin(Dataset):
	def __init__(self, time_series: pd.DataFrame, win_len: int, q_len: int, decompose_law = None):
		super().__init__()
		time_series = torch.tensor(time_series.to_numpy(dtype=np.float32))
		decompose_law = DecompSlidingWin.mean_std_decompose if decompose_law is None else decompose_law

		# Initialize
		Q, Tq = DecompSeries(size=q_len, placeholder=0), DecompSeries(size=q_len, placeholder=-1)
		for t in range(win_len):
			if decompose_law(time_series[: win_len], time_series[t]):
				Q.enqueue(time_series[t])
				Tq.enqueue(t)
		Q.update()
		Tq.update()

		self.windows, self.labels, self.Tx, self.is_anomaly, self.Q, self.Tq = [], [], [], [], [], []

		# fetch and decompose sliding window
		for t in range(win_len, time_series.shape[0]):
			win = time_series[t - win_len: t]
			label = time_series[t]
			is_anomaly = decompose_law(win, label)

			self.windows.append(win)
			self.Tx.append(torch.arange(t - win_len, t, dtype=torch.float))

			self.labels.append(label)
			self.is_anomaly.append(1 if is_anomaly else 0)

			self.Q.append(Q.to_tensor(dtype=torch.float))
			self.Tq.append(Tq.to_tensor(dtype=torch.float))

			if is_anomaly:
				Q.enqueue(label).update()
				Tq.enqueue(t).update()

	def __getitem__(self, idx):
		return self.labels[idx], self.is_anomaly[idx], self.windows[idx], self.Tx[idx], self.Q[idx], self.Tq[idx]

	def __len__(self):
		assert len(self.labels) == len(self.windows) == len(self.Tx) == len(self.Q) == len(self.Tq), \
			(len(self.labels), len(self.windows), len(self.Tx), len(self.Q), len(self.Tq))
		return len(self.labels)

	@staticmethod
	def mean_std_decompose(win: torch.tensor, label, alpha: float = 1.5):
		return (abs(label - win.mean()) / win.std() > alpha).item()
