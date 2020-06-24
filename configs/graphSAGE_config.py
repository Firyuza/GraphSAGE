from datetime import datetime


# custom_aggregator=dict(
#         type='ProteinAggregator',
#         nrof_neigh_per_batch=10,
#         embd_shape=[3, 1],
#         depth=2,
#         aggregators_shape=[(1, 1, 2, 1), (1, 1, 2, 1)],
#         attention_shapes=[(2, 1), (2, 1)],
#         aggregator_type=dict(
#             type='PoolAggregator',
#             activation='leaky_relu',
#             pool_op='reduce_max',
#             # attention_layer=None
#             attention_layer=dict(
#                 type='GATLayer',
#                 attention_mechanism=dict(
#                     type='SingleLayerMechanism'
#                 ),
#                 activation='leaky_relu'
#             )
#         )
#     )

# custom_aggregator=dict(
#         type='ProteinAggregator',
#         nrof_neigh_per_batch=nrof_neigh_per_batch,
#         embd_shape=[3, 1],
#         depth=2,
#         aggregators_shape=[((None, nrof_neigh_per_batch, 1), 1, 2, 1),
#                            ((None, nrof_neigh_per_batch, 1), 1, 2, 1)],
#         attention_shapes=[(2, 1), (2, 1)],
#         aggregator_type=dict(
#             type='RNNAggregator',
#             activation='relu',
#             cell_type='LSTMCell',
#             # attention_layer=None,
#             attention_layer=dict(
#                 type='GATLayer',
#                 attention_mechanism=dict(
#                     type='SingleLayerMechanism'
#                 ),
#                 activation='sigmoid'
#             )
#         )
#     )

# model settings
nrof_neigh_per_batch=10
depth=2

model = dict(
    type='GraphSAGE',
    in_shape=1,
    out_shape=1,
    activation='sigmoid',
    custom_aggregator=dict(
        type='ProteinAggregator',
        nrof_neigh_per_batch=nrof_neigh_per_batch,
        embd_shape=[3, 1],
        depth=depth,
        aggregators_shape=[(2, 1), (2, 1)],
        attention_shapes=[(2, 1), (2, 1)],
        aggregator_type=dict(
            type='MeanAggregator',
            activation='leaky_relu',
            attention_layer=dict(
                type='GATLayer',
                attention_heads=1,
                attention_mechanism=dict(
                    type='SingleLayerMechanism'
                ),
                activation='leaky_relu',
                output_activation='sigmoid'
            )
        )
    ),
    loss_cls=dict(
        type='BinaryCrossEntropyLoss',
        from_logits=False,
        label_smoothing=0,
        loss_weight=1.0),
    accuracy_cls=dict(
        type='BinaryAccuracy'
    )
)
# model training and testing settings
train_cfg = dict(
    reg_loss=dict(
        type='l2_loss',
        weight_decay=0.0005),
    )
test_cfg = dict(
    aggregator_activation='relu')

dataset_type = 'ProteinDataset'
data_root = '/home/firiuza/MachineLearning/'
data = dict(
    train=dict(
        type=dataset_type,
        dataset_name='proteins',
        ann_file=data_root + 'proteins.mat',
        depth=depth,
        nrof_neigh_per_batch=nrof_neigh_per_batch),
    valid=dict(
        type=dataset_type,
        dataset_name='proteins',
        ann_file=data_root + 'proteins.mat'),
    test=dict(
        type=dataset_type,
        dataset_name='proteins',
        ann_file=data_root + 'proteins.mat')
    )

# dataset settings
data_loader_type = 'TensorSlicesDataset'
data_loader_chain_rule = {
    'map': {'num_parallel_calls': 4},
    'batch': {'batch_size': 16},
}
data_loader = dict(
        train=dict(
            type=data_loader_type,
            ops_chain=data_loader_chain_rule,
            map_func_name='prepare_train_data'
        )
)
# learning policy
lr_schedule = dict(
    initial_learning_rate=3e-2,
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

use_TensorBoard=True

# yapf:enable
# runtime settings
total_epochs = 3

log_level = 'INFO'
work_dir = '/home/firiuza/PycharmProjects/GraphSAGE/run_models/run_%s_%s' % (model['custom_aggregator']['aggregator_type']['type'],
                                                                             datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))

restore_model_path = None

workflow = [('train', 2)]#, ('valid', 1)]
