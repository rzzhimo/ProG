import torch
import numpy as np
import torchmetrics
import warnings


class Evaluator:
    def __init__(self, eval_metric='hits@50'):

        self.eval_metric = eval_metric
        if 'hits@' in self.eval_metric:
            self.K = int(self.eval_metric.split('@')[1])

    def _parse_and_check_input(self, input_dict):
        if 'hits@' in self.eval_metric:
            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'

            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            return y_pred_pos, y_pred_neg, type_info

        elif 'mrr' == self.eval_metric:

            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'
            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 2:
                raise RuntimeError('y_pred_neg must to 2-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    def eval(self, input_dict):

        if 'hits@' in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):

        if type_info == 'torch':
            res = torch.topk(y_pred_neg, self.K)
            kth_score_in_negative_edges = res[0][:, -1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {'hits@{}'.format(self.K): hitsK}

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = torch.argsort(y_pred, dim=1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            mrr_list = 1. / ranking_list.to(torch.float)
            return mrr_list.mean()
        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg], axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            mrr_list = 1. / ranking_list.astype(np.float32)
            return mrr_list.mean()


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


def acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device):
    if device == "cpu":
        PG = PG.to("cpu")
        if answering is not None:
            answering = answering.to("cpu")
        gnn = gnn.to("cpu")
    if task_type == "multi_class_classification":
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    elif task_type == "binary_classification":
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    else:
        raise NotImplementedError

    for batch_id, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        if answering:  # if answering is not None

            prompted_graph = PG(test_batch)
            if gnn.gnn_type == "CGCNN":
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.edge_attr, prompted_graph.batch)
            elif gnn.gnn_type == "deeperGAT":
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.edge_attr, prompted_graph.glob_feat, prompted_graph.batch)
            else:
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            # print(graph_emb)
            pre = answering(graph_emb)
        else:  # if answering is None
            if gnn.gnn_type == "CGCNN":
                emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.edge_attr, test_batch.batch)
                pg_batch = PG.token_view()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.edge_attr, pg_batch.batch)
            elif gnn.gnn_type == "deeperGAT":
                emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.edge_attr, test_batch.glob_feat, test_batch.batch)
                pg_batch = PG.token_view()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.edge_attr, pg_batch.glob_feat, pg_batch.batch)
            else:
                emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
                pg_batch = PG.token_view()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))

            # if task_type == 'multi_class_classification':
            pre = torch.softmax(dot, dim=1)
            # elif task_type == 'regression':
            #     pre = torch.sigmoid(dot)
            #     pre = pre.detach()

        pre = pre.detach()
        y = test_batch.y

        pre_cla = torch.argmax(pre, dim=1)
        # print(pre_cla)
        # print(y)

        acc = accuracy(pre_cla, y)
        ma_f1 = macro_f1(pre_cla, y)
        # print("Batch {} Acc: {:.4f} | Macro-F1: {:.4f}".format(batch_id, acc.item(), ma_f1.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    print("Final True Acc: {:.4f} | Macro-F1: {:.4f}".format(acc.item(), ma_f1.item()))
    accuracy.reset()
    macro_f1.reset()
    if device == "cpu":
        PG = PG.to(device)
        if answering is not None:
            answering = answering.to(device)
        gnn = gnn.to(device)
    


def regression_metrics_over_batches(test_loader, PG, gnn, answering, device):
    if device == "cpu":
        PG = PG.to("cpu")
        if answering is not None:
            answering = answering.to("cpu")
        gnn = gnn.to("cpu")

    mse_metric = torchmetrics.MeanSquaredError().to(device)
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    r2_metric = torchmetrics.R2Score().to(device)


    # 存储每个批次的MAE和MSE
    all_mae = []
    all_mse = []
    max_errors = []

    for batch_id, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)

        if answering:
            # Generate prompted graph
            prompted_graph = PG(test_batch)
            # prompted_graph = prompted_graph.to("cpu")
            # Get graph embeddings
            if gnn.gnn_type == "CGCNN":
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.edge_attr, prompted_graph.batch)
            elif gnn.gnn_type == "deeperGAT":
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.edge_attr, prompted_graph.glob_feat, prompted_graph.batch)
            else:
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)

            # Make predictions
            predictions = answering(graph_emb)
        else:
            # Get embeddings for original graphs and prompt tokens
            if gnn.gnn_type == "CGCNN":
                emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.edge_attr, test_batch.batch)
                pg_batch = PG.token_view()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.edge_attr, pg_batch.batch)
            elif gnn.gnn_type == "deeperGAT":
                emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.edge_attr, test_batch.glob_feat, test_batch.batch)
                pg_batch = PG.token_view()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.edge_attr, pg_batch.glob_feat, pg_batch.batch)
            else:
                emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
                pg_batch = PG.token_view()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)

            # Calculate similarity
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            predictions = torch.sigmoid(dot)  # or another suitable activation function

        # Detach predictions and ground truth from computation graph
        predictions = predictions.detach().squeeze()
        ground_truth = test_batch.y.squeeze()

        # print(f"predictions device: {predictions.device}")
        # print(f"ground_truth device: {ground_truth.device}")

        # Update metrics for the current batch
        batch_mse = mse_metric(predictions, ground_truth)
        batch_mae = mae_metric(predictions, ground_truth)
        final_r2  = r2_metric(predictions, ground_truth)

        # 计算最大误差
        batch_max_error = torch.max(torch.abs(predictions - ground_truth))

        # 存储每个批次的MAE和MSE
        all_mae.append(batch_mae)
        all_mse.append(batch_mse)
        max_errors.append(batch_max_error)

        # # Print batch metrics
        # print(f"Batch {batch_id} MSE: {batch_mse.item():.4f} | MAE: {batch_mae.item():.4f}")


    # Finalize metrics
    final_mse = mse_metric.compute()
    final_mae = mae_metric.compute() # 等于mean_mae
    final_r2 = r2_metric.compute()

    # 计算RMSE（均方根误差）
    final_rmse = torch.sqrt(final_mse)

    # 转换为numpy数组以进行额外的计算
    all_mae_np = torch.stack(all_mae).cpu().numpy()
    all_mse_np = torch.stack(all_mse).cpu().numpy()
    max_errors_np = torch.stack(max_errors).cpu().numpy()


    # 计算平均值、标准差和最大误差
    mean_mae = np.mean(all_mae_np)
    std_mae = np.std(all_mae_np)

    mean_rmse = np.mean(np.sqrt(all_mse_np))  # 计算所有批次RMSE的均值

    max_max_error = np.max(max_errors_np)  # 最大的最大误差

    print("final results:")
    # print(f"Final MSE: {final_mse.item():.4f}| Final RMSE: {final_rmse.item():.4f}| Final MAE: {final_mae.item():.4f}")
    print(f"Mean MAE: {mean_mae:.4f}, Final MAE: {final_mae:.4f}")
    print(f"Std MAE: {std_mae:.4f}")
    print(f"Mean RMSE: {mean_rmse:.4f}")
    print(f"Max Max Error: {max_max_error:.4f}")
    print(f"Final R^2: {final_r2.item():.4f}")

    # Reset metrics for next evaluation
    mse_metric.reset()
    mae_metric.reset()
    r2_metric.reset()

    # Move models back to the original device
    if device == "cpu":
        PG = PG.to(device)
        if answering is not None:
            answering = answering.to(device)
        gnn = gnn.to(device)

# Usage Example
# dataloader = DataLoader(...)  # Assume test_loader is defined
# regression_metrics_over_batches(test_loader, PG, gnn, answering, device)
