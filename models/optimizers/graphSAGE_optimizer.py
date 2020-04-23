import tensorflow as tf

from .base_optimizer import BaseOptimizer
from ..registry import OPTIMIZERS

@OPTIMIZERS.register_module
class GraphSAGEOptimizer(BaseOptimizer):
    def __init__(self, optimizer_cfg):
        super(GraphSAGEOptimizer, self).__init__(optimizer_cfg)

        return

    def apply_gradients(self, tape, loss, model):
        grads = tape.gradient(loss, model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return