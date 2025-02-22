import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


def mean_std_stripe(data: pd.DataFrame, n_std: int = 1):
	"""data: (num_experiment, num_feature)"""
	mean = data.mean(axis=0)
	std = data.std(axis=0)

	upper = mean + n_std * std
	lower = mean - n_std * std

	plt.plot(data.columns, mean)
	plt.fill_between(data.columns, lower, upper, alpha=0.4)


def min_max_stripe(data: pd.DataFrame):
	"""data: (num_experiment, num_feature)"""
	mean = data.mean(axis=0)
	max_ = data.max(axis=0)
	min_ = data.min(axis=0)

	plt.plot(data.columns, mean, label='mean')
	plt.plot(data.columns, min_, label='min')
	plt.plot(data.columns, max_, label='max')
	plt.fill_between(data.columns, min_, max_, alpha=0.08)
	plt.legend()
