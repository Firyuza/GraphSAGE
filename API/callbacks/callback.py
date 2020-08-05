from .tensorboard_logger import TensorBoardLogger

class Callback:
    def __init__(self, model, optimizer, tensorboard_logger):
        self.optimizer = optimizer
        self.model = model
        self.tensorboard_logger = tensorboard_logger

        return

    def before_train_epoch(self):
        return

    def after_train_epoch(self):
        return

    def before_train(self):
        return

    def after_train(self):
        return

    def before_train_step(self):
        return

    def after_train_step(self, tape, loss, values_dict, step, mode):
        self.optimizer.apply_gradients(tape, loss, self.model)

        self.tensorboard_logger.log_scalar(mode, 'learning_rate', self.optimizer.get_current_lr(step), step)
        self.tensorboard_logger.log_scalars(mode, values_dict, step)

        return

    def before_valid_epoch(self):
        return

    def after_valid_epoch(self, mode, step, embeddings, labels, predictions):
        self.tensorboard_logger.log_embeddings_by_gt_label(mode, step, embeddings, labels)
        self.tensorboard_logger.log_embeddings_by_max_prediction(mode, step, embeddings, predictions)

        return

    def before_valid(self):
        return

    def after_valid(self):
        return

    def before_valid_step(self):
        return

    def after_valid_step(self, values_dict, step, mode):

        self.tensorboard_logger.log_scalar(mode, 'learning_rate', self.optimizer.get_current_lr(step), step)
        self.tensorboard_logger.log_scalars(mode, values_dict, step)

        return