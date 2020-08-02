# Graph Neural Network

The architecture of this project is inspired by [https://github.com/open-mmlab](https://github.com/open-mmlab)

There are **GraphSAGE**, **GAT** models.


**GraphSAGE**

Inductive Representation Learning on Large Graphs (William L. Hamilton et al.)
![graphSAGE](readme_data/graphSAGE.png)

**GAT**

Graph Attention Networks (Yoshua Bengio et al.)
![GAT](readme_data/GAT.png)


### Dependencies

* Python==3.6+
* tensorflow==
* tensorflow-tensorboard==
* numpy==
* networkx==
* tensorflow_addons==

### Usage

**Config file:**
* In *model* dict define the model architecture
* Define *learning rate* schedule and *optimizer* params
* In *train_cfg* and *test_cfg* define needed settings for training/testing phase that are used in model class.
* Define *dataset_type* for training/validation/testing
* Define *data loader* type and its chain operations that are exist in [tf.data API](https://www.tensorflow.org/guide/data?hl=ru)

**Training:**

For training run *main_train.py* file


### Results for GraphSAGE
Valid *PPI* dataset

Mean Aggregator | Max Pool Aggregator | RNN Aggregator 
----------------| --------------------| --------------
![MeanF1](readme_data/MeanF1.png) | ![MaxPoolF1](readme_data/MaxPoolF1.png) | ![RNNF1](readme_data/RNNF1.png)
Node embeddings for label #23|Node embeddings for label #23 | Node embeddings for label #23
![MeanEmbd](readme_data/MeanEmbd.png) | ![MaxPoolEmbd](readme_data/MaxPoolEmbd.png) | ![RNNEmbd](readme_data/RNNEmbdValid.png)

### Author
[Firiuza](https://medium.com/@Firiuza) medium / [Firyuza](https://github.com/Firyuza) github / [Firiuza](https://twitter.com/Firiuza1) twitter


