import logging
import os.path as osp
import time
import tensorflow as tf

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

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        # self.timestamp = get_time_str()

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

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def train(self, data_loader, **kwargs):
        self.mode = 'train'
        self.data_loader = data_loader

        self.callback.before_train_epoch()
        for i, data_batch in enumerate(data_loader.data_loader):
            self._inner_iter = i
            self.callback.before_train_step()

            with tf.GradientTape() as tape:
                outputs = self.batch_processor(self.model, data_batch, train_mode=True, **kwargs)

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            assert 'total_loss' in outputs

            self.outputs = outputs
            self.callback.after_train_step(tape, outputs['total_loss'], outputs, self.step, self.mode)
            self.step += 1

        self.callback.after_train_epoch()
        self._epoch += 1

        return

    def valid(self, data_loader, **kwargs):
        self.mode = 'valid'
        self.data_loader = data_loader
        self.callback.before_valid_epoch()

        for i, data_batch in enumerate(data_loader.data_loader):
            self._inner_iter = i
            self.callback.before_valid_step()

            outputs = self.batch_processor(self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            self.outputs = outputs
            self.callback.after_valid_step(outputs, self.step, self.mode)

        self.callback.after_valid_epoch()

        return

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.
        """
        self._max_epochs = max_epochs
        self.logger.info('Start running, work_dir: %s' % self.work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                print(mode)

                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        # self.call_hook('after_run')

        return