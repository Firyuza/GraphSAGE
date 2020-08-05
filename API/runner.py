import logging
import os.path as osp
import os
import time
import tensorflow as tf
import h5py

from .callbacks.tensorboard_logger import TensorBoardLogger
from .callbacks.callback import Callback


class Runner(object):
    """A training helper for TesnorFlow.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer,
                 work_dir,
                 logger):
        assert callable(batch_processor)
        self.model = model
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self.restore_model_path = None

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self.tensorboard_logger = TensorBoardLogger(work_dir)
        self.callback = Callback(model, optimizer, self.tensorboard_logger)

        self.logger = logger
        self.work_dir = work_dir
        self.mode = None
        self._epoch = 0
        self.step = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self.step

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def load_checkpoint(self, filename):
        self.restore_model_path = filename
        self.logger.info('load checkpoint from %s', filename)

        file = h5py.File(filename, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)].value)
        self.model.set_weights(weight)

        self.step = int(filename.split('.h5')[0].split('-')[-1])
        self.optimizer.assign_step(self.step)

        return

    def restore_optimizer_parameters(self):
        if self.restore_model_path is not None:
            file = h5py.File(self.restore_model_path.replace('model-', 'optimizer-'), 'r')
            weight = []
            for i in range(len(file.keys())):
                weight.append(file['weight' + str(i)].value)
            self.optimizer.set_weights(weight)

            print('Optimizer parameters restored')

        return

    def save_checkpoint(self, out_dir, filename_tmpl='model-{}.h5'):
        print('Saving variables')
        filename = filename_tmpl.format(self.iter)
        filepath = osp.join(out_dir, 'models')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        save_path = osp.join(filepath, filename)

        file = h5py.File(save_path, 'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

        filename = filename_tmpl.format(self.iter).replace('model', 'optimizer')
        save_path = osp.join(filepath, filename)
        file = h5py.File(save_path, 'w')
        weight = self.optimizer.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

        return

    # @tf.function
    def train(self, data_loader, **kwargs):
        self.mode = 'train'
        self.data_loader = data_loader

        self.callback.before_train_epoch()
        for i, data_batch in enumerate(data_loader.data_loader):
            self._inner_iter = i
            self.callback.before_train_step()

            with tf.GradientTape() as tape:
                outputs, vis_embeddings, batch_labels, predictions = self.batch_processor(self.model, data_batch,
                                                                                          train_mode=True, **kwargs)

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            assert 'total_loss' in outputs

            self.outputs = outputs
            self.callback.after_train_step(tape, outputs['total_loss'], outputs, self.step, self.mode)
            self.step += 1

            if i == 0 and self._epoch == 0:
                self.restore_optimizer_parameters()

        self.save_checkpoint(self.work_dir)
        self.callback.after_train_epoch()
        self._epoch += 1

        return

    def valid(self, data_loader, **kwargs):
        self.mode = 'valid'
        self.data_loader = data_loader
        self.callback.before_valid_epoch()

        all_vis_embeddings = []
        all_labels = []
        all_predictions = []
        for i, data_batch in enumerate(data_loader.data_loader):
            self._inner_iter = i
            self.callback.before_valid_step()

            outputs, vis_embeddings, batch_labels, predictions = self.batch_processor(self.model, data_batch,
                                                                                      train_mode=False)
            all_vis_embeddings.extend(vis_embeddings.numpy())
            all_labels.extend(batch_labels.numpy())
            all_predictions.extend(predictions.numpy())
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            self.outputs = outputs
            self.callback.after_valid_step(outputs, self.step, self.mode)

        self.callback.after_valid_epoch(self.mode, self.step, all_vis_embeddings,
                                        all_labels, predictions)

        return

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.
        """
        self._max_epochs = max_epochs
        self.logger.info('Start running, work_dir: %s' % self.work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        while self.epoch < max_epochs:
            print('Epoch %d' % self.epoch)
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                print(mode)

                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)

        return