import torch
from ProG.utils import seed, seed_everything
seed_everything(seed)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import numpy as np
from ProG.eva import Evaluator
import warnings
from collections import defaultdict
import pickle as pk
from random import shuffle
from torch_geometric.data import Batch

def multi_class_data(dataname, num_class):
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        data_path1 = 'Dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = 'Dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:100]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = 'Dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = 'Dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:100]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print(key, value)

    return train_data, test_data

def mrr_hit(normal_label: np.ndarray, pos_out: np.ndarray, metric: list = None):
    if isinstance(normal_label, np.ndarray) and isinstance(pos_out, np.ndarray):
        pass
    else:
        warnings.warn('it would be better if normal_label and out are all set as np.ndarray')

    results = {}
    if not metric:
        metric = ['mrr', 'hits']

    if 'hits' in metric:
        hits_evaluator = Evaluator(eval_metric='hits@50')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        for N in [100]:
            neg_test_pred_N = neg_test_pred.view(-1, 100)
            for K in [1, 5, 10]:
                hits_evaluator.K = K
                test_hits = hits_evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred_N,
                })[f'hits@{K}']

                results[f'Hits@{K}@{N}'] = test_hits

    if 'mrr' in metric:
        mrr_evaluator = Evaluator(eval_metric='mrr')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        neg_test_pred = neg_test_pred.view(-1, 100)

        mrr = mrr_evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })

        if isinstance(mrr, torch.Tensor):
            mrr = mrr.item()
        results['mrr'] = mrr
    return results

def eva(pre, label, task_type='multi_class_classification'):
    if task_type == 'regression':
        mae = mean_absolute_error(label, pre)
        mse = mean_squared_error(label, pre)
        return {"mae": mae, "mse": mse}
    elif task_type == 'multi_class_classification':
        pre_cla = torch.argmax(pre, dim=1)
        acc = accuracy_score(label, pre_cla)
        mac_f1 = f1_score(label, pre_cla, average='macro')
        mic_f1 = f1_score(label, pre_cla, average='micro')
        return {"acc": acc, "mac_f1": mac_f1, "mic_f1": mic_f1}
    elif task_type == 'link_prediction':
        normal_label = label
        pos_out = pre[:, 1]
        results = mrr_hit(normal_label, pos_out)
        return results
    else:
        raise NotImplemented(
            "eva() function is currently only used for multi-class classification  and link_prediction tasks!")
