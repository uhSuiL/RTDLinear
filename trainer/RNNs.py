import dataset as ds
import os
import torch
import pandas as pd

from dataset import ClusterDataset
from model import GRU, LSTM
from . import Base, Logger
from dataset.util import TimeFeature, train_valid_test, prepare_time_series

from torch import nn
from torch.utils.data import Dataset
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError


class RNNTrainer(Base):
	def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
		df = ds.util.load_clustered_data(
			cluster_info='./data/cluster_info/cluster_info_a_s.csv',
			processed_data='./data/processed/data.csv',
		)[self.train_config.cluster_id]

		cluster_datasets = {'train': [], 'valid': [], 'test': []}
		for series_id in df.columns:
			series_data, self.num_time_feat = prepare_time_series(df, series_id)
			train_data, valid_data, test_data = train_valid_test(
				series_data,
				train_ratio=0.7,
				test_ratio=0.2,
			)
			cluster_datasets['train'].append(ds.SlidingWinDataset(train_data, sliding_win=self.train_config.sliding_win))
			cluster_datasets['valid'].append(ds.SlidingWinDataset(valid_data, sliding_win=self.train_config.sliding_win))
			cluster_datasets['test'].append(ds.SlidingWinDataset(test_data, sliding_win=self.train_config.sliding_win))
		return (
			ClusterDataset(cluster_datasets['train']),
			ClusterDataset(cluster_datasets['valid']),
			ClusterDataset(cluster_datasets['test']),
		)

	def customize_sample(self, standard_sample) -> tuple:
		cluster_id, (y_wins, X_wins) = standard_sample

		y_wins = y_wins.unsqueeze(dim=-2)  # expand dim: num_series = 1
		y = y_wins[:, :, : -self.num_time_feat]
		cluster_id = torch.concat([cluster_id.unsqueeze(dim=-1)] * self.train_config.sliding_win, dim=-1)  #.to(dtype=torch.float)
		return [X_wins, cluster_id], y

	@staticmethod
	def train_gru(train_config: dict = None, *, cluster_id = None, save_dir = './log/'):
		RNNTrainer(
			Model=GRU.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir=save_dir if cluster_id is None else os.path.join(save_dir, f'cluster{cluster_id}'),
			logger=Logger(),
			model_config='./config/GRU.yml',
			train_config={
				'cluster_id': 0 if cluster_id is None else cluster_id,
				'sliding_win': 20,
				'device': 'cuda',

				'dataloader': {
					'batch_size': 32,
					'shuffle': False,
				},
				'optimizer': {
					'lr': 0.004
				}
			} if train_config is None else train_config,
		). \
			train(num_epoch=400)

	@staticmethod
	def train_lstm(train_config: dict = None, *, cluster_id = None, save_dir = './log/'):
		RNNTrainer(
			Model=LSTM.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir=save_dir if cluster_id is None else os.path.join(save_dir, f'cluster{cluster_id}'),
			logger=Logger(),
			model_config='./config/LSTM.yml',
			train_config={
				'cluster_id': 0 if cluster_id is None else cluster_id,
				'sliding_win': 20,
				'device': 'cuda',

				'dataloader': {
					'batch_size': 32,
					'shuffle': False,
				},
				'optimizer': {
					'lr': 0.01
				}
			} if train_config is None else train_config,
		). \
			train(num_epoch=400)

	@staticmethod
	def test_gru(experiment_time, epoch, cluster_id: int = None):
		RNNTrainer(
			Model=GRU.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log/' if cluster_id is None else f'./log/cluster{cluster_id}/',
			logger=Logger(),
			resume={
				'trainer time': experiment_time,
				'epoch': epoch,
			},
		). \
			test()

	@staticmethod
	def test_lstm(experiment_time, epoch, cluster_id: int = None):
		RNNTrainer(
			Model=LSTM.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log/' if cluster_id is None else f'./log/cluster{cluster_id}/',
			logger=Logger(),
			resume={
				'trainer time': experiment_time,
				'epoch': epoch,
			},
		). \
			test()


class RNNClusterTrainer(Base):
	def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
		total_clustered_dataset = {'train': [], 'valid': [], 'test': []}
		for cluster_id in pd.read_csv(self.train_config.cluster_info_path)['agg clusters']:

			df = ds.util.load_clustered_data(
				cluster_info=self.train_config.cluster_info_path,
				processed_data='./data/processed/data.csv',
			)[cluster_id]

			cluster_datasets = {'train': [], 'valid': [], 'test': []}
			for series_id in df.columns:
				series_data, self.num_time_feat = prepare_time_series(df, series_id)
				train_data, valid_data, test_data = train_valid_test(
					series_data,
					train_ratio=0.7,
					test_ratio=0.2,
				)
				cluster_datasets['train'].append(ds.SlidingWinDataset(train_data, sliding_win=self.train_config.sliding_win))
				cluster_datasets['valid'].append(ds.SlidingWinDataset(valid_data, sliding_win=self.train_config.sliding_win))
				cluster_datasets['test'].append(ds.SlidingWinDataset(test_data, sliding_win=self.train_config.sliding_win))

			total_clustered_dataset['train'].append(ClusterDataset(cluster_datasets['train'], return_ds_id=False))
			total_clustered_dataset['valid'].append(ClusterDataset(cluster_datasets['valid'], return_ds_id=False))
			total_clustered_dataset['test'].append(ClusterDataset(cluster_datasets['test'], return_ds_id=False))

		return (
			ClusterDataset(total_clustered_dataset['train']),
			ClusterDataset(total_clustered_dataset['valid']),
			ClusterDataset(total_clustered_dataset['test']),
		)

	def customize_sample(self, standard_sample) -> tuple:
		cluster_id, (y_wins, X_wins) = standard_sample

		y_wins = y_wins.unsqueeze(dim=-2)  # expand dim: num_series = 1
		y = y_wins[:, :, : -self.num_time_feat]
		cluster_id = torch.concat([cluster_id.unsqueeze(dim=-1)] * self.train_config.sliding_win, dim=-1)
		return [X_wins, cluster_id], y

	@staticmethod
	def train_gru(train_config: dict = None):
		RNNClusterTrainer(
			Model=GRU.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log_cluster/',
			logger=Logger(),
			model_config='./config/GRU_cluster.yml',
			train_config={
				'cluster_info_path': './data/cluster_info/cluster_info_a_s.csv',
				'sliding_win': 20,
				'device': 'cuda',

				'dataloader': {
					'batch_size': 32,
					'shuffle': False,
				},
				'optimizer': {
					'lr': 0.004
				}
			} if train_config is None else train_config,
		). \
			train(num_epoch=800)

	@staticmethod
	def train_gru2(train_config: dict = None):
		RNNClusterTrainer(
			Model=GRU.Model2,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log_cluster/',
			logger=Logger(),
			model_config='./config/GRU_cluster.yml',
			train_config={
				'cluster_info_path': './data/cluster_info/cluster_info_a_s.csv',
				'sliding_win': 20,
				'device': 'cuda',

				'dataloader': {
					'batch_size': 32,
					'shuffle': False,
				},
				'optimizer': {
					'lr': 0.001
				}
			} if train_config is None else train_config,
		). \
			train(num_epoch=800)

	@staticmethod
	def train_lstm(train_config: dict = None):
		RNNClusterTrainer(
			Model=LSTM.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log_cluster/',
			logger=Logger(),
			model_config='./config/LSTM_cluster.yml',
			train_config={
				'cluster_info_path': './data/cluster_info/cluster_info_a_s.csv',
				'sliding_win': 20,
				'device': 'cuda',

				'dataloader': {
					'batch_size': 32,
					'shuffle': False,
				},
				'optimizer': {
					'lr': 0.001
				}
			} if train_config is None else train_config,
		). \
			train(num_epoch=800)

	@staticmethod
	def train_lstm2(train_config: dict = None):
		RNNClusterTrainer(
			Model=LSTM.Model2,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log_cluster/',
			logger=Logger(),
			model_config='./config/LSTM_cluster.yml',
			train_config={
				'cluster_info_path': './data/cluster_info/cluster_info_a_s.csv',
				'sliding_win': 20,
				'device': 'cuda',

				'dataloader': {
					'batch_size': 32,
					'shuffle': False,
				},
				'optimizer': {
					'lr': 0.001
				}
			} if train_config is None else train_config,
		). \
			train(num_epoch=800)

	@staticmethod
	def test_gru(experiment_time, epoch):
		RNNClusterTrainer(
			Model=GRU.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(),
						  MeanAbsolutePercentageError()],
			save_dir='./log_cluster/',
			logger=Logger(),
			resume={
				'trainer time': experiment_time,
				'epoch': epoch,
			},
		). \
			test()

	@staticmethod
	def test_lstm(experiment_time, epoch):
		RNNClusterTrainer(
			Model=LSTM.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(),
						  MeanAbsolutePercentageError()],
			save_dir='./log_cluster/',
			logger=Logger(),
			resume={
				'trainer time': experiment_time,
				'epoch': epoch,
			},
		). \
			test()
