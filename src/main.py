import tensorflow as tf
import numpy as np
import os
from gensim.models.keyedvectors import KeyedVectors

data_path = os.path.abspath('..') + '/data/'
# we use gensim(a useful tool) to get embedding vector, due to the privacy protection,
# we only provide the embedding data
twitter_embedding_path = data_path + 'twitter_embedding.emb'
twitter_vocab_path = data_path + 'twitter_model.vocab'
foursquare_embedding_path = data_path + 'foursquare_embedding.emb'
foursquare_vocab_path = data_path + 'foursquare_model.vocab'
# there are 3148 anchor users in data set, we use 2098 users for training, 1050 users for testing
connect_data_path = data_path + 'trainConnect.txt'
connect_test_data_path = data_path + 'testConnect.txt'
# in this simplified  version, we will train our model directly
# connect_warm_up_data_path = data_path + 'trainConnect_400_warm_up.txt'
# our embedding data size is 800
embedding_size = 800
# load the embedding vector using gensim
x_vectors = KeyedVectors.load_word2vec_format(foursquare_embedding_path, binary=False, fvocab=foursquare_vocab_path)
y_vectors = KeyedVectors.load_word2vec_format(twitter_embedding_path, binary=False, fvocab=twitter_vocab_path)
inputs = []     # train input vector
labels = []     # train label vector
test_inputs = []  # test input vectors
test_labels = []  # test label words


def load_data():
    f = open(connect_data_path)
    for line in f.readlines():
        line_array = line.strip().split(' ')
        if line_array[0] not in x_vectors.vocab.keys() or line_array[1] not in y_vectors.vocab.keys():
            print("======================warning!!!" + line_array[0] + " or " + line_array[
                1] + "does not exsits!!!=====================================")
            continue
        inputs.append(x_vectors[line_array[0]])
        labels.append(y_vectors[line_array[1]])
    print('input size:' + str(len(inputs)))
    print('labels size:' + str(len(labels)))


# this function can be replace by tf.dense in a higher tensorflow version
def add_layer(input_data, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(input_data, weights) + biases
    if activation_function is None:
        outputs = result
    else:
        outputs = activation_function(result)
    return outputs


# record the current data index
data_index = 0


# note that: len(inputs) == len(labels)
def generate_batch(type):
    """
    get the batch data
    :param type: train or test
    :return: batch data
    """
    global data_index
    if type == 'train':
        if data_index + batch_size >= len(inputs):  # the case that now_index + batch_size > total data
            batch_inputs = inputs[data_index:]
            batch_labels = labels[data_index:]
            data_index = batch_size - len(batch_inputs)
            for d in inputs[:data_index]:
                batch_inputs.append(d)
            for l in labels[:data_index]:
                batch_labels.append(l)
        else:
            batch_inputs = inputs[data_index:data_index + batch_size]
            batch_labels = labels[data_index:data_index + batch_size]
            data_index += batch_size
        return batch_inputs, batch_labels
    elif type == 'test':
        f = open(connect_test_data_path)
        for line in f.readlines():
            line_array = line.strip().split(' ')
            if line_array[0] not in x_vectors.vocab.keys() or line_array[1] not in y_vectors.vocab.keys():
                print("======================warning!!!" + line_array[0] + " or " + line_array[
                    1] + "does not exsits!!!=====================================")
                continue
            test_inputs.append(x_vectors[line_array[0]])
            test_labels.append(line_array[1])
        print('test_inputs size:' + str(len(test_inputs)))
        print('test_labels size:' + str(len(test_labels)))
        return test_inputs


def normalize_vector(vector):
    norm = tf.sqrt(tf.reduce_sum(tf.square(vector), 1, keep_dims=True))
    normalized_embeddings = vector / norm
    return normalized_embeddings


# get the Levenshtein distance
def leven_dis(str1, str2):
    len_str1 = len(str1.lower()) + 1
    len_str2 = len(str2.lower()) + 1
    # create matrix
    matrix = [0 for n in range(len_str1 * len_str2)]
    # init x axis
    for i in range(len_str1):
        matrix[i] = i
    # init y axis
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]


def rank(topn, target):
    result = []
    for item in topn:
        max_length = len(item[0]) if len(item[0]) > len(target) else len(target)
        modify_value = ((max_length / 2.0 - leven_dis(item[0], target) * 1.0) / (max_length / 2.0)) * 0.05
        val = item[1] + modify_value
        if val > 1.0:
            val = 1.0
        if val < 0:
            val = 0
        result.append((item[0], val))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result


# build net
xs = tf.placeholder(tf.float32, [None, embedding_size])
ys = tf.placeholder(tf.float32, [None, embedding_size])
hidden_1 = add_layer(xs, embedding_size, 1200, None)
output_x = add_layer(hidden_1, 1200, embedding_size, None)
results = tf.matmul(normalize_vector(ys), normalize_vector(output_x), transpose_b=True)
loss_x = 1 - tf.reduce_mean(tf.diag_part(results))
train_step_x = tf.train.GradientDescentOptimizer(1).minimize(loss_x)
init = tf.global_variables_initializer()
num_steps = 2000001
# due to the data is not big, we set a small batch size
batch_size = 1

with tf.Session() as session:
    print("program begin")
    init.run()
    load_data()
    batch_size = 1
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch('train')
        feed_dict = {xs: batch_inputs, ys: batch_labels}
        loss_val, _ = session.run([loss_x, train_step_x], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss_x at step ", step, ": ", average_loss)
            average_loss = 0
        if step % 20000 == 0:
            test_inputs = []
            test_labels = []
            prediction = session.run(output_x, feed_dict={xs: generate_batch('test')})
            count = 0
            total = np.zeros(101)
            for vector in prediction:
                number_in_topn = 0
                topn = y_vectors.similar_by_vector(vector=vector, topn=100)
                rank_result = rank(topn, test_labels[count])
                for item in rank_result:
                    number_in_topn += 1
                    if item[0] == test_labels[count]:
                        index = number_in_topn
                        while index < 101:
                            total[index] += 1
                            index += 1
                count += 1
            for i in range(1, 101):
                if i in [1, 5, 10, 15, 30, 100]:
                    print('top ' + str(i) + ' : ' + str(total[i] / count))
