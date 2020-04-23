import numpy as np
import tensorflow as tf
import os

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))
        self.valid_summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'valid'))

        self.writers = {
            'train': self.train_summary_writer,
            'valid': self.valid_summary_writer
        }

        return

    def restore_model(self, model_path):
        return

    def save_model(self, model, save_path):
        return

    def log_scalar(self, mode, name, value, step=None):
        with self.writers[mode].as_default():
            tf.summary.scalar(name, value, step=step)

        return

    def log_scalars(self, mode, values_dict, step):
        with self.writers[mode].as_default():
            for name, value in values_dict.items():
                if value.shape == ():
                    tf.summary.scalar(name, value, step=step)

        return