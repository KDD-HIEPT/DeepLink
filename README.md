# DeepLink
a simplified python implementation for DeepLink [DeepLink: A Deep Learning Approach for User Identity Linkage](https://ieeexplore.ieee.org/abstract/document/8486231)

## Requirements
- python >= 2.7
- tensorflow >= 1.3.0
- numpy >= 1.14.0
- gensim >= 3.4.0

## Datasets
The dataset used in this project are from [Aligning users across social
networks using network embedding](https://www.ijcai.org/Proceedings/16/Papers/254.pdf). Due to the privacy concern, we do not provide all the raw data. However, the embedding data and some train data can be found in the ` data`  folder.


## Usage
To run DeepLink, first clone the project to your python IDE (eg:Pycharm), then run the `main.py`.
Our embedding method is introduced in the file `embedding.py`, which is a `random_walk` and `word2vec` implementation .
>Note: you need to install the required libs.

