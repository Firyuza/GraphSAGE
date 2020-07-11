import tensorflow as tf

from abc import ABCMeta, abstractmethod

class BaseGraph(tf.keras.models.Model, metaclass=ABCMeta):
    """Base class for Graph"""

    def __init__(self):
        super(BaseGraph, self).__init__()

    @abstractmethod
    def reset_model_states(self):
        pass

    @abstractmethod
    def build_model(self, *args):
        pass

    def build(self, *args):

        self.build_model(*args)

        super(BaseGraph, self).build(())

    @abstractmethod
    def call_accuracy(self, *args):
        """

        :param args:
        :return:
        """
        return

    @abstractmethod
    def call_loss(self, *args):
        """

        :param args:
        :return:
        """
        return

    @abstractmethod
    def call_train(self, *args):
        """

        :param graph_nodes:
        :param labels:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def call_test(self, *args):
       pass

    def call(self, *args, train_mode):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if train_mode:
            return self.call_train(*args)
        else:
            return self.call_test(*args)