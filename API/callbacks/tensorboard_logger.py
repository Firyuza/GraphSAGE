import numpy as np
import tensorflow as tf
import os

from tensorboard.plugins import projector

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

    def log_embeddings_by_gt_label(self, mode, step, embeddings, labels):
        # Set up a logs directory, so Tensorboard knows where to look for files
        log_dir = os.path.join(self.log_dir, 'graph_embeddings/by_gt_labels_%s_%d_step/' % (mode, step))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save Labels separately on a line-by-line manner.
        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            for label in labels:
                if len(np.where(label == 1)[0]) > 0:
                    str_label = str(np.where(label == 1)[0][0] if len(np.where(label == 1)[0]) == 1
                                    else np.where(label == 1)[0][1])
                else:
                    str_label = '-1'
                f.write("{}\n".format(str_label))

        # Save the weights we want to analyse as a variable. Note that the first
        # value represents any unknown word, which is not in the metadata, so
        # we will remove that value.
        weights = tf.Variable(embeddings)
        # Create a checkpoint from embedding, the filename and key are
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir, config)

        return

    def log_embeddings_by_max_prediction(self, mode, step, embeddings, predictions):
        # Set up a logs directory, so Tensorboard knows where to look for files
        log_dir = os.path.join(self.log_dir, 'graph_embeddings/by_predictions_%s_%d_step/' % (mode, step))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save Labels separately on a line-by-line manner.
        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            for pred in predictions:
                str_label = str(np.argmax(pred))
                f.write("{}\n".format(str_label))

        # Save the weights we want to analyse as a variable. Note that the first
        # value represents any unknown word, which is not in the metadata, so
        # we will remove that value.
        weights = tf.Variable(embeddings)
        # Create a checkpoint from embedding, the filename and key are
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir, config)

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