import torch
import pickle as pk
from random import shuffle

from torch_geometric.data import Batch
from collections import defaultdict
from .utils import restore_data


def multi_class_NIG(dataname, num_class,shots=100):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list,test_list

def regression_matbench(dataname, shots=100, train_rate=0):
    """
    Load matbench_dielectric dataset for regression tasks.
    
    :param dataname: Name of the dataset, e.g. 'matbench_dielectric'
    :param shots: Number of samples for training set
    :return: train_data, test_data, train_list, test_list
    """
    # Load the dataset from .pkl file
    # with open('Dataset/{}/{}.pkl'.format(dataname, dataname), 'rb') as file:
    #     dataset = pk.load(file)

    # Load the dataset from .pt file
    data, slice = torch.load('Dataset/{}/{}.pt'.format(dataname, dataname))
    dataset = restore_data(data, slice)

    # Shuffle the dataset
    shuffle(dataset)

    if train_rate == 0:
        train_list = dataset[:shots]
        test_list = dataset[shots:shots*10]
    else:
        train_shots = int(len(dataset)*train_rate)
        train_list = dataset[:train_shots]
        test_list = dataset[train_shots:]

    # Convert lists of Data objects into Batch objects
    train_data = Batch.from_data_list(train_list)
    test_data = Batch.from_data_list(test_list)

    print("Training set: {} graphs".format(len(train_list)))
    print("Testing set: {} graphs".format(len(test_list)))

    return train_data, test_data, train_list, test_list

    
def binary_class_matbench(dataname, shots=100, train_rate=0):
    """
    Load matbench_dielectric dataset for class tasks.
    
    :param dataname: Name of the dataset, e.g. 'matbench_dielectric'
    :param shots: Number of samples for training set
    :return: train_data, test_data, train_list, test_list
    """
    # Load the dataset from .pkl file
    data, slice = torch.load('Dataset/{}/{}.pt'.format(dataname, dataname))
    dataset = restore_data(data, slice)

    # Shuffle the dataset
    shuffle(dataset)

    if train_rate == 0:
        train_list = dataset[:shots]
        test_list = dataset[shots:shots*10]
    else:
        train_shots = int(len(dataset)*train_rate)
        train_list = dataset[:train_shots]
        test_list = dataset[train_shots:]

    # Convert lists of Data objects into Batch objects
    train_data = Batch.from_data_list(train_list)
    test_data = Batch.from_data_list(test_list)

    print("Training set: {} graphs".format(len(train_list)))
    print("Testing set: {} graphs".format(len(test_list)))

    return train_data, test_data, train_list, test_list

if __name__ == '__main__':
    pass
