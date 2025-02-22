import dataset as ds
import os
import torch
import pandas as pd

from dataset import ClusterDataset
from model import (
	TDLinear,
	RTDLinear,
)
from trainer import Base, Logger
from dataset.util import train_valid_test, prepare_time_series

from torch import nn
from torch.utils.data import Dataset
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError


class DLinearPlusTrainer(Base):
	def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
		df = ds.util.load_clustered_data(
			cluster_info='./data/cluster_info/cluster_info_a_s.csv',
			processed_data='./data/processed/data.csv',
		)[self.train_config.cluster_id]

		cluster_datasets = {'train': [], 'valid': [], 'test': []}
		for series_id in df.columns:
			series_data, self.num_time_feat = prepare_time_series(df, series_id)
			series_data = series_data.reset_index().rename(columns={'index': 'time_ticks'})

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
		y = y_wins[..., 1: -self.num_time_feat]  # exclude the time ticks(col=0)
		X = X_wins[..., :-self.num_time_feat]
		return [X], y

	@staticmethod
	def train_td_linear(train_config = None, *, cluster_id = None, save_dir='./log/'):
		DLinearPlusTrainer(
			Model=TDLinear.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,

			# Scheduler=torch.optim.lr_scheduler.StepLR,
			# scheduler_config={'step_size': 80, 'gamma': 0.2},

			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir=save_dir if cluster_id is None else os.path.join(save_dir, f'cluster{cluster_id}'),
			logger=Logger(),
			model_config='./config/TDLinear.yml',
			train_config={
				'cluster_id': 0 if cluster_id is None else cluster_id,
				'sliding_win': 25,
				'device': 'cuda',

				'dataloader': {
					'batch_size': 32,
					'shuffle': False,
				},

				'optimizer': {
					'lr': 0.004,
				}
			} if train_config is None else train_config,
		). \
			train(num_epoch=2000)

	@staticmethod
	def train_td_linear_resume(experiment_time, epoch, cluster_id):
		DLinearPlusTrainer(
			Model=TDLinear.Model,
			loss_fn= nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log/' if cluster_id is None else f'./log/cluster{cluster_id}/',
			logger=Logger(),
			resume={
				'trainer time': experiment_time,
				'epoch': epoch,
			},
		). \
			train(num_epoch=200)

	@staticmethod
	def train_rtd_linear(train_config = None, *, cluster_id = None):
		DLinearPlusTrainer(
			Model=RTDLinear.Model,
			loss_fn=nn.MSELoss(),
			Optimizer=torch.optim.Adam,
			metrics_list=[MeanSquaredError(), MeanSquaredError(squared=False), MeanAbsoluteError(), MeanAbsolutePercentageError()],
			save_dir='./log/' if cluster_id is None else f'./log/cluster{cluster_id}/',
			logger=Logger(),
			model_config='./config/TDLinear.yml',
			train_config={
				'cluster_id': 0 if cluster_id is None else cluster_id,
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
			train(num_epoch=400)

	@staticmethod
	def test_td_linear(experiment_time, epoch, cluster_id: int = None):
		DLinearPlusTrainer(
			Model=TDLinear.Model,
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
	def test_rtd_linear(experiment_time, epoch, cluster_id: int = None):
		DLinearPlusTrainer(
			Model=RTDLinear.Model,
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
