from argparse import ArgumentParser

import pandas as pd

from trainer import (
	RNNs,
	DLinear,
	DLinearPlus,
)
from trainer.DLinearPlus import DLinearPlusTrainer

if __name__ == '__main__':
	parser = ArgumentParser(description='Choose Model Experiment')
	parser.add_argument('--model', type=str, default='GRU')

	args = parser.parse_args()

	cluster_info = pd.read_csv('./data/cluster_info/cluster_info_a_s.csv')

	if args.model == 'LSTM':
		experiment_times = ['2024-10-05-08-54', '2024-10-05-09-14', '2024-10-05-09-24', '2024-10-05-09-27']
		use_epochs = [143, 272, 4, 87]
		for cluster_id in cluster_info['agg clusters']:
			RNNs.RNNTrainer.test_lstm(experiment_times[cluster_id], use_epochs[cluster_id],  cluster_id=cluster_id)

	elif args.model == 'GRU':
		experiment_times = ['2024-10-05-09-51', '2024-10-05-10-01', '2024-10-05-10-11', '2024-10-05-10-13']
		use_epochs = [195, 9, 178, 10]
		for cluster_id in cluster_info['agg clusters']:
			RNNs.RNNTrainer.test_gru(experiment_times[cluster_id], use_epochs[cluster_id], cluster_id=cluster_id)

	elif args.model == 'DLinear':
		experiment_times = ['2024-10-05-10-25', '2024-10-05-10-36', '2024-10-05-10-43', '2024-10-05-10-45']
		use_epochs = [18, 48, 26, 22]
		for cluster_id in cluster_info['agg clusters']:
			DLinear.DLinearTrainer.test_dlinear(experiment_times[cluster_id], use_epochs[cluster_id], cluster_id=cluster_id)

	elif args.model == 'TDLinear':
		experiment_times = ['2024-10-05-11-02', '2024-10-05-11-21', '2024-10-05-11-34', '2024-10-05-11-37']
		use_epochs = [79, 15, 107, 74]
		for cluster_id in cluster_info['agg clusters']:
			DLinearPlus.DLinearPlusTrainer.test_td_linear(experiment_times[cluster_id], use_epochs[cluster_id], cluster_id=cluster_id)

	elif args.model == 'RTDLinear':
		experiment_times = ['2024-10-05-11-53', '2024-10-05-12-12', '2024-10-05-12-25', '2024-10-05-12-28']
		use_epochs = [328, 214, 68, 259]
		for cluster_id in cluster_info['agg clusters'][3:]:
			DLinearPlusTrainer.test_rtd_linear(experiment_times[cluster_id], use_epochs[cluster_id], cluster_id=cluster_id)
