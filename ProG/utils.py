import os
import numpy as np
import random
import torch
from copy import deepcopy
from random import shuffle
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
import pickle as pk
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch import nn, optim
import torch.nn.functional as F
import pickle

seed = 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


# used in pre_train.py
def gen_ran_output(data, model):
    vice_model = deepcopy(model)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    if model.gnn.gnn_type == "CGCNN":
        z2 = vice_model.forward_cl_cgcnn(data.x, data.edge_index, data.edge_attr, data.batch)
    elif model.gnn.gnn_type == "deeperGAT":
        z2 = vice_model.forward_cl_deeperGAT(data.x, data.edge_index, data.edge_attr, data.glob_feat, data.batch)
    else:
        z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)

    return z2


# used in pre_train.py
def load_data4pretrain(dataname='CiteSeer', num_parts=200):
    data = pk.load(open('./Dataset/{}/feature_reduced.data'.format(dataname), 'br'))
    print(data)

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    input_dim = data.x.shape[1]
    hid_dim = input_dim
    graph_list = list(ClusterData(data=data, num_parts=num_parts, save_dir='./Dataset/{}/'.format(dataname)))

    # 输出图的个数
    print(f"图的个数：{len(graph_list)}")

    # 输出前5个图
    for i in range(5):
        print(graph_list[i])
    return graph_list, input_dim, hid_dim

def restore_data(data, slices):
    restored_data_list = []

    # 假设所有图都有节点特征x
    # 根据x属性的分片信息，计算图的数量
    num_graphs = slices['x'].size(0) - 1

    for i in range(num_graphs):
        single_data_dict = {}
        
        # 注意这里的修改：使用data.keys()而不是data.keys
        for key in data.keys():
            item = slices[key]
            # 为每个属性获取对应的分片范围
            start, stop = item[i].item(), item[i + 1].item()

            # 特别处理edge_index
            if key == 'edge_index':
                edge_index = data[key][:, start:stop]
                # 调整edge_index为相对索引
                edge_index = edge_index - edge_index.min()
                single_data_dict[key] = edge_index
            else:
                # 处理其他属性
                single_data_dict[key] = data[key][start:stop]

        restored_data_list.append(Data(**single_data_dict))
    
    return restored_data_list


# used in pre_train.py,load matbench data
def load_mbdata4pretrain(dataname):
    # with open('./Dataset/{}/{}.pkl'.format(dataname,dataname), 'rb') as file:
    #     dataset = pk.load(file)
    data, slice = torch.load('./Dataset/{}/{}.pt'.format(dataname,dataname))
    dataset = restore_data(data, slice)

    # with open(f'./Dataset/{dataname}/{dataname}.pkl', "wb") as file:
    #     pickle.dump(dataset, file)
    #     print(f'Dataset {dataname} saved.')
    # dataset 是一个包含多个 Data 对象的列表
    # 计算输入维度和隐藏层维度
    input_dim = dataset[0].x.shape[1]  # 假设所有图的特征维度都相同
    hid_dim = 100  # 你可以根据需要调整隐藏层维度
    
    # 输出图的个数
    print(f"图的个数：{len(dataset)}")
    
    # 输出前5个图
    for i in range(5):
        print(dataset[i])
    
    return dataset, input_dim, hid_dim

# used in prompt.py
def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        if x is None:
            return torch.nn.LeakyReLU()
        else:
            return F.leaky_relu(x)
    elif act_type == 'tanh':
        if x is None:
            return torch.nn.Tanh()
        else:
            return torch.tanh(x)






















def __seeds_list__(nodes):
    split_size = max(5, int(nodes.shape[0] / 400))
    seeds_list = list(torch.split(nodes, split_size))
    if len(seeds_list) < 400:
        print('len(seeds_list): {} <400, start overlapped split'.format(len(seeds_list)))
        seeds_list = []
        while len(seeds_list) < 400:
            split_size = random.randint(3, 5)
            seeds_list_1 = torch.split(nodes, split_size)
            seeds_list = seeds_list + list(seeds_list_1)
            nodes = nodes[torch.randperm(nodes.shape[0])]
    shuffle(seeds_list)
    seeds_list = seeds_list[0:400]

    return seeds_list


def __dname__(p, task_id):
    if p == 0:
        dname = 'task{}.meta.train.support'.format(task_id)
    elif p == 1:
        dname = 'task{}.meta.train.query'.format(task_id)
    elif p == 2:
        dname = 'task{}.meta.test.support'.format(task_id)
    elif p == 3:
        dname = 'task{}.meta.test.query'.format(task_id)
    else:
        raise KeyError

    return dname


def __pos_neg_nodes__(labeled_nodes, node_labels, i: int):
    pos_nodes = labeled_nodes[node_labels[:, i] == 1]
    pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
    neg_nodes = labeled_nodes[node_labels[:, i] == 0]
    neg_nodes = neg_nodes[torch.randperm(neg_nodes.shape[0])]
    return pos_nodes, neg_nodes


def __induced_graph_list_for_graphs__(seeds_list, label, p, num_nodes, potential_nodes, ori_x, same_label_edge_index,
                                      smallest_size, largest_size):
    seeds_part_list = seeds_list[p * 100:(p + 1) * 100]
    induced_graph_list = []
    for seeds in seeds_part_list:

        subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=1, num_nodes=num_nodes,
                                         edge_index=same_label_edge_index, relabel_nodes=True)

        temp_hop = 1
        while len(subset) < smallest_size and temp_hop < 5:
            temp_hop = temp_hop + 1
            subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=temp_hop, num_nodes=num_nodes,
                                             edge_index=same_label_edge_index, relabel_nodes=True)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.from_numpy(np.setdiff1d(potential_nodes.numpy(), subset.numpy()))

            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            # directly downmsample
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
            subset = torch.unique(torch.cat([torch.flatten(seeds), subset]))

        sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)

        x = ori_x[subset]
        graph = Data(x=x, edge_index=sub_edge_index, y=label)
        induced_graph_list.append(graph)

    return induced_graph_list


def graph_views(data, aug='random', aug_ratio=0.1):
    if aug == 'dropN':
        data = drop_nodes(data, aug_ratio)

    elif aug == 'permE':
        data = permute_edges(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio)
    elif aug == 'random':
        n = np.random.randint(2)
        if n == 0:
            data = drop_nodes(data, aug_ratio)
        elif n == 1:
            data = permute_edges(data, aug_ratio)
        else:
            print('augmentation error')
            assert False
    return data


def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
    #               (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    
    # 初始设为空列表，用于存储更新后的边索引
    new_edge_index = []
    # 如果存在边属性，同样设为空列表，用于存储更新后的边属性
    new_edge_attr = [] if data.edge_attr is not None else None
    for n in range(edge_num):
        if edge_index[0, n] not in idx_drop and edge_index[1, n] not in idx_drop:
            # 仅当两个节点都未被删除时，保留该边
            new_edge_index.append([idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]])
            if new_edge_attr is not None:
                # 同步更新边属性
                new_edge_attr.append(data.edge_attr[n])

    try:
        data.edge_index = torch.tensor(new_edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        if new_edge_attr is not None:
            data.edge_attr = torch.stack(new_edge_attr)

        # 对节点位置pos进行更新
        if data.pos is not None:
            data.pos = data.pos[idx_nondrop]

        # 对全局特征glob_feat进行更新
        if data.glob_feat is not None:
            data.glob_feat = data.glob_feat[idx_nondrop]
    except Exception as e:
        print(f"Error in drop_nodes: {e}")
        data = data

    return data


def permute_edges(data, aug_ratio):
    """
    only change edge_index, all the other keys unchanged and consistent
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]

    # 如果存在edge_attr，则同样只保留选中的边对应的属性
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[idx_delete]

    return data


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token.clone().detach()

    return data
