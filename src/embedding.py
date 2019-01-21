# -*- coding:  UTF-8 -*-
from numpy import *
import random
import argparse
import os
import time
from gensim.models import word2vec


data_path = os.path.dirname(os.getcwd()) + '/data/'
# the social graph file path,if user1 and user2 are friends, there will be a line as "user1\tuser2" in the file
account_data_path_x = data_path + 'twitter.following'
# the path that records the random walk results
walk_file_x = data_path + 'walk_all_x.txt'
# the folder path to records the embedding results
embedding_path = data_path + 'embedding/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_paths', default='10')
    args_ = parser.parse_args()
    return args_


def print_msg(msg=''):
    if not msg.strip():
        msg = 'print_msg call'
    print('============================================================================================================')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '  ' + msg)
    print('============================================================================================================')


def load_graph_dict(file_name):
    data_dict = {}
    fx = open(file_name)
    for line in fx.readlines():
        line_arr = line.replace('\n', '').split('\t')
        length = len(line_arr)
        if length < 2:
            p(line_arr[0] + 'not paired,will skip!!!')
            break
        if line_arr[0] not in data_dict.keys():
            data_dict[str(line_arr[0])] = [str(line_arr[1])]
        if line_arr[1] not in data_dict[str(line_arr[0])]:
            data_dict[str(line_arr[0])].append(str(line_arr[1]))
    return data_dict


# walk begin
def walk(graph_dict, num_paths, walk_file):
    print_msg("walk for " + walk_file + " start ")
    walk_list = []  # each item in this list is a random walk result,such as : user1,user2,user3...
    f1 = open(walk_file, 'w+')
    for j in range(int(num_paths)):  # the total loop times
        nodes = list(graph_dict.keys())
        random.Random(0).shuffle(nodes)  # shuffle nodes
        for key_start_node in nodes:  # wo ensure every node will have a random walk result
            if len(graph_dict[key_start_node]) < 1:
                continue
            for first_neighbor in graph_dict[key_start_node]:
                walk_list.append(key_start_node)
                walk_list.append(first_neighbor)
                random_walk(graph_dict, walk_list, first_neighbor, 2)
                for e in walk_list:
                    f1.write('%s ' % (e))
                f1.write('\n')
                walk_list = []
    f1.close()
    print_msg("walk for " + walk_file + " complete ")


# Recursive call random_walk to get random walk results until walk length is reached
def random_walk(graph_dict, walk_list, now_node, now_length, walk_length=40):
    if now_length >= walk_length: # walk length is reached
        return 0
    if now_node not in graph_dict.keys() or len(graph_dict[now_node]) < 1: # no more next node
        return 0
    now_node = random.Random().choice(graph_dict[now_node])
    walk_list.append(now_node)
    random_walk(graph_dict, walk_list, now_node, now_length + 1)


def start_walk(num_paths):
    # x_dict["user1"] = [user2, user3, user4] means that user2, user3, user4 are user1's friends
    x_dict = load_graph_dict(account_data_path_x)
    # random walk begin
    walk(x_dict, num_paths, walk_file=walk_file_x)
    print_msg("random walk has complete!")


def start_embedding():
    walkList_x = word2vec.LineSentence(walk_file_x)
    print_msg('embedding x start!!!')
    modelX = word2vec.Word2Vec(walkList_x, negative=10, sg=1, hs=0, size=100, window=4, min_count=0, workers=15, iter=30)
    # save the embedding results
    modelX.wv.save_word2vec_format(embedding_path + 'twitter.emb', fvocab=embedding_path + 'twitter.vocab')
    print_msg('embedding x end!!!')

if __name__ == "__main__":
    num_path = parse_args().num_paths
    print_msg("gonna random walk "+num_path+" times!!!")
    start_walk(num_path)
    start_embedding()