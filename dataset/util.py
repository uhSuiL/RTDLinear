import pandas as pd
import torch


def train_valid_test(data: pd.DataFrame, *, train_ratio: float, test_ratio: float) -> tuple:
	"""
	For TSF task, we construct
		- former train_ratio of data as train_data
		- latter test_ratio of data as test_data
		- midst left data as valid_data
	along time-axis

	:return train_data, valid_data, test_data
	"""
	data = data.reset_index(drop=True)
	T = data.shape[0]

	train_data = data.iloc[: int(T * train_ratio), :]
	test_data = data.iloc[-int(T * test_ratio):, :]
	valid_data = data.drop(train_data.index).drop(test_data.index)
	return train_data, valid_data, test_data


def get_period_dummies(time_len: int, period_list: list[int], phase: int = 0) -> pd.DataFrame:
	period_dummies = []
	for p in period_list:
		ticks = [(t + phase) % p for t in range(time_len)]
		dummies = pd.get_dummies(ticks, prefix=f'period_{p}')
		assert dummies.shape[-1] == p, f'{dummies.shape} != {p}'
		period_dummies.append(dummies)
	return pd.concat(period_dummies, axis=1)


class TimeFeature:
	@staticmethod
	def day_of_week(ticks: pd.Series) -> pd.Series:
		"""day of week projected to [-0.5, 0.5]"""
		return ((ticks.dt.dayofweek / 6) - 0.5).rename('DoW')

	@staticmethod
	def day_of_month(ticks: pd.Series) -> pd.Series:
		"""day of month projected to [-0.5, 0.5]"""
		return ((ticks.dt.day - 1) / 30 - 0.5).rename('DoM')

	@staticmethod
	def day_of_year(ticks: pd.Series) -> pd.Series:
		"""day of year projected to [-0.5, 0.5]"""
		return ((ticks.dt.year - 1) / 365 - 0.5).rename('DoY')

	@staticmethod
	def week_of_year(ticks: pd.Series) -> pd.Series:
		"""week of year projected to [-0.5, 0.5]"""
		return ((ticks.dt.isocalendar().week - 1) / 52 - 0.5).rename('WoY')

	@staticmethod
	def month_of_year(ticks: pd.Series) -> pd.Series:
		"""month of year projected to [-0.5, 0.5]"""
		return ((ticks.dt.month - 1) / 11 - 0.5).rename('MoY')


def load_clustered_data(cluster_info: str, processed_data: str) -> list[pd.DataFrame]:
	cluster_info = pd.read_csv(cluster_info)
	processed_data = pd.read_csv(processed_data, index_col=0)
	cluster_info['index'] = cluster_info['index'].apply(eval)
	return [
		processed_data[c]
		for c in cluster_info['index']
	]


def prepare_time_series(df: pd.DataFrame, series_id, drop_date = True, return_n_time_feat = True):
	"""
	1. get date ticks ids, drop date col optionally
	2. concat time features
	"""
	df = df[series_id] \
		.reset_index(drop=False) \
		.rename(columns={'index': 'date'})

	df['date'] = pd.to_datetime(df['date'])
	time_features = pd.concat([
		TimeFeature.day_of_week(df['date']),
		TimeFeature.day_of_month(df['date']),
		TimeFeature.day_of_year(df['date']),
		TimeFeature.week_of_year(df['date']),
		TimeFeature.month_of_year(df['date']),
	], axis=1)

	ts = pd.concat([df, time_features], axis=1)

	if drop_date:
		ts = ts.drop('date', axis=1)

	if return_n_time_feat:
		return ts, time_features.shape[-1]
	return ts


class DecompSeries:
	"""
	- fixed size
	- automatically deque when enqueue
	- replace placeholder after update
	"""

	def __init__(self, size: int, placeholder = None):
		self.size = size
		self.queue = [placeholder] * size
		self.__last_blank_idx = size - 1  # only effective when non-negative

	def __len__(self):
		assert self.size == len(self.queue)
		return len(self.queue)

	def __getitem__(self, idx):
		return self.queue[idx]

	def __repr__(self):
		return str(self.queue)

	@property
	def last_blank_idx(self):
		return self.__last_blank_idx

	def enqueue(self, value):
		self.queue.append(value)
		self.queue.pop(0)
		self.__last_blank_idx -= 1
		return self

	def enqueue_batch(self, values):
		self.queue += values
		self.queue = self.queue[-self.size:]
		self.__last_blank_idx -= len(values)
		return self

	def update(self):
		"""
		pseudo code:
			q_set <- set(q).remove(placeholder)
			new_q <- q_set
			ptr <- 1
			while new_q.length < size:
				new_q.insert(0, q_set[-ptr])
				ptr <- (ptr + 1) % q_set.length
		"""
		q_set = list(set(self.queue))
		if self.__last_blank_idx >= 0 and len(q_set) > 1:
			q_set.remove(self.queue[self.__last_blank_idx])
			self.__last_blank_idx = -1

		new_q = q_set.copy()
		for i in range(1, 1 + self.size - len(q_set)):
			new_q.insert(0, q_set[-(i % len(q_set))])

		self.queue = new_q
		return self

	def to_tensor(self, **kwargs):
		return torch.tensor(self.queue, **kwargs)
