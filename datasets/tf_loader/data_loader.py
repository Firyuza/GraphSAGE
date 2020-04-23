import tensorflow as tf

from ..registry import DATA_LOADER

@DATA_LOADER.register_module
class TensorSlicesDataset:
    def __init__(self, dataset, ops_chain, map_func_name):
        # TODO make asssertion that data fits to from_tensor_slices
        self.dataset = dataset

        data_slices = self.dataset.prepare_train_indices()
        self.data_loader = tf.data.Dataset.from_tensor_slices((data_slices))

        self.__set_up_chain(ops_chain, map_func_name)

        return

    def get_dataset(self):
        return self.dataset

    def __set_up_chain(self, input_ops_chain, map_func_name):
        for key, values in input_ops_chain.items():
            if 'map' in key and map_func_name is not None:
                values['map_func'] = getattr(self.dataset, map_func_name)
            getattr(self, key)(values)

        return

    def map(self, kwargs):
        self.data_loader = self.data_loader.map(**kwargs)

        return

    def batch(self, kwargs):
        self.data_loader = self.data_loader.batch(**kwargs)

        return