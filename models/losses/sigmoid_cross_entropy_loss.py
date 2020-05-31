import tensorflow as tf

from ..registry import LOSSES


@LOSSES.register_module
class SigmoidCrossEntropyLoss(tf.keras.losses.Loss):

    def __init__(self, loss_weight=1.0):
        super(SigmoidCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

        self.cls_criterion = tf.nn.sigmoid_cross_entropy_with_logits

    def call(self, labels, preds):
        loss_cls = self.loss_weight * \
                   tf.reduce_mean(
                       tf.reduce_mean(self.cls_criterion(labels=tf.cast(labels, dtype=tf.float32), logits=preds),
                                     axis=1))

        return loss_cls