from datetime import datetime

# aggregator_layers=dict(
#         type='PPIAggregator',
#         nrof_neigh_per_batch=nrof_neigh_per_batch,
#         depth=depth,
#         aggregators_shape=[(50, 50), (100, 50)],
#         attention_shapes=[(50, 50, 1), (100, 50, 1)],
#         aggregator_type=dict(
#             type='MeanAggregator',
#             activation='leaky_relu',
#             use_concat=True,
#             attention_layer=None
#             # attention_layer=dict(
#             #     type='GATLayer',
#             #     attention_heads=1,
#             #     attention_mechanism=dict(
#             #         type='SingleLayerMechanism'
#             #     ),
#             #     activation='leaky_relu',
#             #     output_activation='sigmoid'
#             # )
#         ),
#     )

# aggregator_layers=dict(
#         type='PPIAggregator',
#         nrof_neigh_per_batch=nrof_neigh_per_batch,
#         depth=depth,
#         aggregators_shape=[(50, 50, 100, 50), (100, 50, 100, 50)],
#         attention_shapes=[(50, 50, 1), (100, 50, 1)],
#         aggregator_type=dict(
#             type='PoolAggregator',
#             activation='leaky_relu',
#             pool_op='reduce_max',
#             use_concat=True,
#             attention_layer=None
#             # attention_layer=dict(
#             #     type='GATLayer',
#             #     attention_heads=1,
#             #     attention_mechanism=dict(
#             #         type='SingleLayerMechanism'
#             #     ),
#             #     activation='leaky_relu',
#             #     output_activation='sigmoid'
#             # )
#         )
# )


# model settings
nrof_neigh_per_batch=25
depth=2
num_classes = 121

model = dict(
    type='GraphSAGE',
    in_shape=100,
    out_shape=num_classes,
    activation='sigmoid',
    aggregator_layers=dict(
            type='PPIAggregator',
            nrof_neigh_per_batch=nrof_neigh_per_batch,
            depth=depth,
            aggregators_shape=[((None, nrof_neigh_per_batch, 50), 50, 50, 50),
                               ((None, nrof_neigh_per_batch, 100), 100, 100, 50)],
            attention_shapes=[(50, 50, 1), (100, 100, 1)],
            aggregator_type=dict(
                type='RNNAggregator',
                activation='relu',
                cell_type='LSTMCell',
                cell_params=dict(
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    dropout=0.0,
                    recurrent_dropout=0.0
                ),
                use_concat=True,
                attention_layer=None
                # attention_layer=dict(
                #     type='GATLayer',
                #     attention_heads=1,
                #     attention_mechanism=dict(
                #         type='SingleLayerMechanism'
                #     ),
                #     activation='leaky_relu',
                #     output_activation='sigmoid'
                # )
            )
    ),
    loss_cls=dict(
        type='BinaryCrossEntropyLoss',
        loss_weight=1.0),
    accuracy_cls=dict(
        type='F1Score',
        num_classes=num_classes,
        average='micro',
        threshold=0.5
    )
)

# learning policy
lr_schedule = dict(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.99,
    staircase=True)
# optimizer
optimizer = dict(
    type='GraphSAGEOptimizer',
    optimizer_cfg=dict(
        type='Adam',
        params=None,
        lr_schedule_type='ExponentialDecay',
        lr_schedule=lr_schedule)
)

# model training and testing settings
train_cfg = dict(
    reg_loss=dict(
        type='l2_loss',
        weight_decay=0.0005),
    )
test_cfg = dict(
    aggregator_activation='relu')

# dataset
dataset_type = 'PPIDataset'
data_root = '/home/firiuza/MachineLearning/ppi/'
data = dict(
    train=dict(
        type=dataset_type,
        dataset_name='ppi',
        ann_file=data_root + 'train_ppi.pickle',
        depth=depth,
        nrof_neigh_per_batch=nrof_neigh_per_batch
    ),
    valid=dict(
        type=dataset_type,
        dataset_name='ppi',
        ann_file=data_root + 'valid_ppi.pickle'),
    test=dict(
        type=dataset_type,
        dataset_name='ppi',
        ann_file=data_root + 'test_ppi.pickle')
    )

# dataset settings
data_loader_type = 'TensorSlicesDataset'
data_loader_chain_rule = {
    'map': {'num_parallel_calls': 4},
    'batch': {'batch_size': 2},
}
data_loader = dict(
        train=dict(
            type=data_loader_type,
            ops_chain=data_loader_chain_rule,
            map_func_name='prepare_train_data'
        )
)

# yapf:enable
# runtime settings
total_epochs = 5000

log_level = 'INFO'
work_dir = '/home/firiuza/models/GraphModels/run_models/run_ppi_%s_%s' % (model['aggregator_layers']['aggregator_type']['type'],
                                                                             datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))

restore_model_path = '/home/firiuza/models/GraphModels/run_models/run_ppi_RNNAggregator_20200718-171553/models/model-5460.h5'
    # '/home/firiuza/models/GraphModels/run_models/run_ppi_PoolAggregator_20200718-180225/models/model-6880.h5'
    # '/home/firiuza/models/GraphModels/run_models/run_ppi_MeanAggregator_20200718-182848/models/model-6910.h5'

workflow = [('valid', 1)] #('train', 1),
