import tensorflow as tf

from ..registry import LOSSES


@LOSSES.register_module
class BinaryCrossEntropyLoss(tf.keras.losses.Loss):

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 loss_weight=1.0):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

        self.cls_criterion = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,
                                                                label_smoothing=label_smoothing)

    def call(self, labels, preds):
        loss_cls = self.loss_weight * \
                   self.cls_criterion( labels, preds)

        return loss_cls
