import os
import csv
import yaml
import random
import shutil
import logging
import argparse

import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from datetime import datetime


class Base:
	"""Basic Entity of Experiment

	Experiment Pipeline:
		train_loader, valid_loader, test_loader

		model, model_config, model_ckpt

	Features
		- Initialize model from model config
		- Support Resuming from ckpt
		- Manage logging and saving
		- Template for model training, validating
		- Template for model testing

	Directory Structure
		`save_dir`/
			`model_name`/
				`trainer time`/
					log.txt
					model_config.yml
					train_config.yml

					train_metrics.csv
					test_metrics.csv

					(external_ckpt.pth)
					checkpoints/
						epoch_{i}.pth
						...

	Initialization Flow
		0. set random seed
		1. set directory structure: create `trainer time/` `{}_metrics.csv` `checkpoints/`
		2. set logger: create `log.txt`
		3. parse resume <if>: load and save model_config, load ckpt, init model, create `resume.yml`, set `start_epoch`
		4. load and save external_ckpt <if>
		5. load model_config, init model and `start epoch`<if>
		6. load data to dataloaders

	Resume Training
		resume = {
			'trainer time': ...,
			'epoch': ...,
		}
		1. If you want to resume from trainer irrelevant ckpt, pass it to `external_ckpt` while `model_config` is required
			the passed external_ckpt will be saved to `checkpoints dir` for others' reproduction
			the processed trainer will be created with processed exp time
		2. If you want to resume from ckpt in an trainer, specify `exp time` and `epoch` (default last epoch)
			the `resume.yml` recording the meta info of resuming will be created and saved
			the `model_config` `model checkpoints` will be loaded from specified exp and epoch and saved again
			the actual trainer time with its dir will be processed

	Format of `{}_metrics.csv`
		- header included
		- {trainer}: epoch_num, loss_name, valid_metrics1_name, valid_metrics2_name, ...
		- {test}: y_true, y_pred, loss_name, valid_metrics1_name, valid_metrics2_name, ...

	Model Implementation Standard
		TODO WRITE
	"""
	def __init__(self,
				 Model,
				 loss_fn,
				 Optimizer: torch.optim.Optimizer,
				 metrics_list: list,

				 save_dir: str,
				 logger: logging.Logger,
				 Scheduler: torch.optim.lr_scheduler.LRScheduler = None,
				 scheduler_config: dict = None,

				 train_config: str | dict = None,
				 model_config: str = None,
				 resume: bool | dict = False,
				 seed = 0,
				 external_ckpt: str = None):
		self.is_loaded = {'model_config': False, 'ckpt': False, 'train_config': False}

		self.seed = seed
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

		self.loss_fn = loss_fn
		self.metrics_fns = metrics_list

		self.logger = logger

		# Setting Directory Structure
		# ===========================
		self.model_dir = os.path.join(save_dir, name(Model, target='module'))
		self.save_dir = os.path.join(
			self.model_dir,
			datetime.now().strftime('%Y-%m-%d-%H-%M')
		)  # save_dir/model_name/exp_time/

		if os.path.exists(self.save_dir):
			shutil.rmtree(self.save_dir)
			self.log(f"Path {self.save_dir} not empty. Content removed", logging.WARNING)

		self.ckpt_dir = os.path.join(self.save_dir, 'checkpoints')
		os.makedirs(self.ckpt_dir)

		self.train_metrics_csv_path = os.path.join(self.save_dir, 'train_metrics.csv')
		with open(self.train_metrics_csv_path, 'w', newline='') as f:
			csv_writer = csv.writer(f)
			header = ['epoch', name(loss_fn), *[name(metrics_fn) for metrics_fn in metrics_list]]
			csv_writer.writerow(header)

		self.test_metrics_csv_path = os.path.join(self.save_dir, 'test_metrics.csv')
		with open(self.test_metrics_csv_path, 'w', newline='') as f:
			csv_writer = csv.writer(f)
			header = ['y_true', 'y_pred', name(loss_fn), *[name(metrics_fn) for metrics_fn in metrics_list]]
			csv_writer.writerow(header)

		#  Setting Logger
		#  ===================================
		log_file = os.path.join(self.save_dir, 'log.txt')
		file_handler = logging.FileHandler(log_file)
		file_formatter = logging.Formatter(LOG_FORMAT)
		file_handler.setFormatter(file_formatter)
		self.logger.addHandler(file_handler)

		_, resume_epoch = self.__parse_resume(resume, Model)
		self.start_epoch = resume_epoch + 1

		# If you want to specify model_config, resume shouldn't be triggered and nothing should have been loaded yet
		if model_config is not None:
			assert not self.is_loaded['model_config'], "Model Config Conflict: model config has been loaded"
			assert not self.is_loaded['ckpt'], "Checkpoint Conflict: checkpoint has been loaded before default init"
			model_config = load_config(model_config)
			self.model = Model(model_config)
			self.model_config = model_config
			self.is_loaded['model_config'] += 1

		if external_ckpt is not None:
			assert not self.is_loaded['ckpt'], "Checkpoint Conflict: checkpoint has been loaded before external ckpt"
			self.model.load_state_dict(torch.load(external_ckpt))
			torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'external_ckpt.pth'))
			self.is_loaded['ckpt'] += 1

		if train_config is not None:
			assert not self.is_loaded['train_config'], "Experiment Config Conflict: experiment config has been loaded"
			self.train_config = load_config(train_config)
			self.is_loaded['train_config'] += 1

		assert self.is_loaded['model_config'] == 1, f"Model config not loaded properly. Load times: {self.is_loaded['model_config']}"
		assert self.is_loaded['ckpt'] <= 1, f"Checkpoint not loaded properly. Load times: {self.is_loaded['ckpt']}"
		assert self.is_loaded['train_config'] == 1, f'Train config not loaded properly. Load times: {self.is_loaded['train_config']}'
		assert self.model_config is not None and self.model is not None and self.train_config is not None

		model_config_file = os.path.join(self.save_dir, 'model_config.yml')
		with open(model_config_file, 'w', encoding='utf-8') as f:
			yaml.dump(vars(self.model_config), f)
			self.log(f"Model config was dumped to {model_config_file}")

		if 'device' not in self.train_config.__dict__.keys():
			self.train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		train_config_file = os.path.join(self.save_dir, 'train_config.yml')
		with open(train_config_file, 'w', encoding='utf-8') as f:
			yaml.dump(vars(self.train_config), f)
			self.log(f"Train Config was dumped to {train_config_file}")

		self.model = self.model.to(self.train_config.device)
		self.metrics_fns = [fn.to(self.train_config.device) for fn in self.metrics_fns]

		self.optimizer = Optimizer(self.model.parameters(), **self.train_config.optimizer)
		self.scheduler = Scheduler(self.optimizer, **scheduler_config) if Scheduler is not None else None

		train_set, valid_set, test_set = self.load_data()
		self.train_loader = DataLoader(train_set, **self.train_config.dataloader, pin_memory=True)
		self.valid_loader = DataLoader(valid_set, **self.train_config.dataloader, pin_memory=True)
		self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

		self.log(f"Epoch checkpoints will be saved to {self.ckpt_dir}")
		self.log(f"Loss Function: {name(loss_fn)}")
		self.log(f"Metrics: {[name(metrics_fn) for metrics_fn in metrics_list]}")
		self.log("====== Experiment Initialization Finished ======")

	def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
		""":return: train_loader, valid_loader, test_loader"""
		raise NotImplementedError

	def __parse_resume(self, resume: dict | bool, Model) -> tuple:
		"""
		Execute
			1. reconstruct `resume` to dict
			2. get `resume_exp_time` and `resume_epoch`
			3. load model config and initialize model
			4. load ckpt to model and log
			5. dump {'trainer time': `resume_exp_time`, 'epoch': `resume_epoch`} to `resume.yml`

		`resume`
			- `True` | `{}`: resume from last epoch of last trainer time
			- `{'trainer time': ...}`: resume from last epoch of given trainer time
			- `{'trainer time': ..., 'epoch': ...}: resume from given epoch of given trainer time`

		Return
			resume trainer time, resume epoch
		"""
		assert not (self.is_loaded['model_config'] and self.is_loaded['ckpt']), self.is_loaded.keys()

		if type(resume) is bool and not resume:
			return None, -1  # 0 for start epoch
		resume = {} if type(resume) is bool and resume else resume
		resume_exp_time: str = resume.get('trainer time', self.find_last_experiment())
		resume_epoch: int = resume.get('epoch', self.find_last_epoch(resume_exp_time))
		resume_model_config = load_config(os.path.join(self.model_dir, resume_exp_time, 'model_config.yml'))

		self.model = Model(resume_model_config)
		self.model_config = resume_model_config
		self.is_loaded['model_config'] = True

		self.train_config = load_config(os.path.join(self.model_dir, resume_exp_time, 'train_config.yml'))
		self.is_loaded['train_config'] = True

		self.load_epoch_ckpt(epoch=resume_epoch, experiment_time=resume_exp_time, log=True)

		with open(os.path.join(self.save_dir, 'resume.yml'), mode='w') as f:
			yaml.dump({
				'trainer time': resume_exp_time,
				'epoch': resume_epoch
			}, f)
		return resume_exp_time, resume_epoch

	def __init_model_from_config(self, Model, model_config: argparse.Namespace):  # TODO CHECK USAGE
		self.model = Model(model_config)
		self.log(f"Model {name(self.model, target='module')} initialized from config")

		new_config_path = os.path.join(self.save_dir, 'model_config.yml')
		with open(new_config_path) as f:
			yaml.dump(vars(model_config), f)
			self.log(f"Model config is saved to {new_config_path}")

		self.is_loaded['model_config'] += 1

	def find_last_experiment(self) -> str:
		exp_times = os.listdir(self.model_dir)
		exp_times = sorted(datetime.strptime(exp_time, '%Y-%m-%d-%H-%M') for exp_time in exp_times)
		return exp_times[-1].strftime('%Y-%m-%d-%H-%M')

	def find_last_epoch(self, exp_time: str) -> int:
		epoch_ckpts = os.listdir(os.path.join(self.model_dir, exp_time, 'checkpoints/'))
		epoch_nums = sorted([int(epoch_ckpt.split('.')[0].split('_')[-1]) for epoch_ckpt in epoch_ckpts])
		return epoch_nums[-1]

	def log(self, msg, level = logging.INFO):
		"""Using `log` instead of `print` during trainer"""
		self.logger.log(level, msg)

	def save_epoch_ckpt(self, epoch: int, experiment_time: str = None, log: bool = False):
		ckpt_dir = self.ckpt_dir if experiment_time is None \
			else os.path.join(self.model_dir, experiment_time, 'checkpoints')
		torch.save(
			self.model.state_dict(),
			os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
		)
		if log:
			self.log(f"Saved ckpt to {ckpt_dir}")

	def load_epoch_ckpt(self, epoch: int, experiment_time: str = None, log: bool = False):
		ckpt_dir = self.ckpt_dir if experiment_time is None \
			else os.path.join(self.model_dir, experiment_time, 'checkpoints')
		self.model.load_state_dict(
			torch.load(os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))
		)
		if log:
			self.log(f"Loaded ckpt from {ckpt_dir}")
		self.is_loaded['ckpt'] += 1

	def customize_sample(self, standard_sample) -> tuple:
		""" Customize samples from loader for specific model
		:param standard_sample: samples generated directly from data_loader/data_set
		:return: (custom_input, y)
		"""
		*X, y = standard_sample
		return X, y

	def train_epoch(self) -> list:
		epoch_loss = []
		for b, batch_samples in enumerate(self.train_loader):
			model_inputs, y_true = self.customize_sample(batch_samples)
			try:
				model_inputs = [model_input.to(self.train_config.device) for model_input in model_inputs]
			except Exception as e:
				print(e)
				print(b)
				print(model_inputs[-1])
				print([(model_input.dtype, model_input.device, model_input.shape) for model_input in model_inputs])
			y_true = y_true.to(self.train_config.device)

			y_pred = self.model(*model_inputs)
			loss = self.loss_fn(y_pred, y_true)

			with torch.no_grad():
				epoch_loss.append(loss)

			self.optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.5)
			self.optimizer.step()

		valid_metrics = self.valid(self.valid_loader, add_loss=False)
		valid_metrics = torch.tensor(valid_metrics).mean(dim=0).tolist()  # final result is the avg among valid batches
		assert len(self.metrics_fns) <= len(valid_metrics) <= len(self.metrics_fns) + 1, \
			f'{len(valid_metrics)}!={len(self.metrics_fns)}'
		return torch.mean(torch.tensor(epoch_loss)), valid_metrics

	def train(self, num_epoch: int):
		"""Utils for epoch training, saving, logging, resuming.

				Decouple utils from model training and validating
		Print
			Epoch {e}/{total_epoch} ({duration: .2f}s): {loss: .4f} | {metrics1, metrics2, ..., : .4f}

		Save
			epoch ckpt -> `epoch_{e}.pth`
			epoch result <{e}, loss, metrics> -> [line e] `train_metrics.csv`
		"""
		for e in range(self.start_epoch, self.start_epoch + num_epoch):

			if self.loss_fn.__class__.__name__ == 'EpochAwareLoss':
				self.loss_fn.epoch = e

			loss, valid_metrics = self.train_epoch()  # float, list[float]

			if self.scheduler is not None:
				self.scheduler.step()
				# self.log(f"Learning rate updated: group0 lr={self.optimizer.param_groups[0]['lr']}")

			self.log(f"Epoch {e + 1}/{num_epoch} | Loss:{loss: .4f} | Metrics:" + ','.join(f'{m: .4f}' for m in valid_metrics))
			self.save_epoch_ckpt(e)
			with open(self.train_metrics_csv_path, mode='a', newline='') as file:
				csv_writer = csv.writer(file)
				csv_writer.writerow([e + 1, loss, *valid_metrics])
		return self

	@torch.no_grad()
	def valid(self, data_loader: DataLoader, *, add_loss: bool = True, y_recorder: list = None) -> list[np.array]:
		metrics_results = []  # (num_batch, num_metrics)
		for b, batch_samples in enumerate(data_loader):
			model_inputs, y_true = self.customize_sample(batch_samples)
			model_inputs = [model_input.to(self.train_config.device) for model_input in model_inputs]
			y_true = y_true.to(self.train_config.device)

			y_pred = self.model(*model_inputs)

			metrics_results.append([
				metrics_fn(y_pred, y_true)
				for metrics_fn in self.metrics_fns
			])

			if add_loss:
				loss = self.loss_fn(y_true, y_pred)
				metrics_results[-1].insert(0, loss)  # loss first

			if y_recorder is not None:  # ATTENTION: Beware of the shape
				y_recorder.append([y_true.item(), y_pred.item()])

		return metrics_results  # (num_batch, num_metrics)

	@torch.no_grad()
	def test(self):
		self.log('====== Start Test ======')
		self.log(f'Total test samples: {len(self.test_loader.dataset)}')
		y_true_pred_list: list[list] = []  # list[list[tensor, tensor]]
		test_metrics = self.valid(self.test_loader, y_recorder=y_true_pred_list)
		assert len(test_metrics) == len(y_true_pred_list), (len(test_metrics), len(y_true_pred_list))

		self.log('====== Start writing result =====')
		with open(self.test_metrics_csv_path, mode='a', newline='') as f:
			csv_writer = csv.writer(f)
			for i in range(len(test_metrics)):
				line = y_true_pred_list[i] + [tsr.item() for tsr in test_metrics[i]]
				csv_writer.writerow(line)

			result = torch.tensor(y_true_pred_list)
			csv_writer.writerow(
				['', '', '']
				+ [
					metrics_fn(result[..., 0], result[..., 1]).item()
					for metrics_fn in self.metrics_fns
				]
			)

		self.log('===== Finish Test =====')
		return self


LOG_FORMAT = '[%(levelname)s] %(asctime)s> %(message)s'


def Logger(log_file: str = None, name: str = None) -> logging.Logger:
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)

	console_handler = logging.StreamHandler()
	console_formatter = logging.Formatter(LOG_FORMAT)
	console_handler.setFormatter(console_formatter)
	logger.addHandler(console_handler)

	if log_file is not None:
		file_handler = logging.FileHandler(log_file)
		file_formatter = logging.Formatter(LOG_FORMAT)
		file_handler.setFormatter(file_formatter)
		logger.addHandler(file_handler)

	return logger


def load_config(yml: str | dict) -> argparse.Namespace:
	assert type(yml) is str or type(yml) is dict, type(yml)
	if type(yml) is str:
		with open(yml, mode='r') as f:
			config = yaml.safe_load(f)
	elif type(yml) is dict:
		config = yml
	else:
		raise RuntimeError(f"Illegal type for yml: {type(yml)}")

	config = argparse.Namespace(**config)
	return config


def name(Obj, *, target: str = 'class') -> str:
	if target == 'class':
		return Obj.__class__.__name__
	elif target == 'module':
		return Obj.__module__.split('.')[-1]
	else:
		raise RuntimeError("Illegal target to name")
