import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv, CGConv, global_add_pool
from torch_geometric.data import Batch, Data
from .utils import act
import warnings
from deprecated.sphinx import deprecated
from torch_geometric.utils import sort_edge_index, add_self_loops, to_undirected
from torch_geometric.loader.cluster import ClusterData
import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        # 定义 MLP 层
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.Softplus(), #使用Softplus激活函数,这是可以换的
            torch.nn.Linear(out_dim, out_dim)
        )

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)

        # Apply MLP encoding
        x = self.mlp(x)

        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb


# 这一部分参考matdeeplearning的代码
class CGCNN(torch.nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='CGCNN'):
        super(CGCNN, self).__init__()

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_node_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        
        self.embedding = torch.nn.Linear(input_node_dim, hid_dim)
        self.embedding_edge = torch.nn.Linear(input_edge_dim, hid_dim)
        hid_dim_edge = hid_dim


        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()

        for i in range(gcn_layer_num):
            self.conv_layers.append(CGConv(hid_dim, hid_dim_edge, aggr="mean", batch_norm=False))
            self.bn_layers.append(torch.nn.BatchNorm1d(hid_dim))

        self.conv_to_fc = torch.nn.Linear(hid_dim, hid_dim)
        self.conv_to_fc_softplus = torch.nn.Softplus()

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.embedding(x)
        edge_attr = self.embedding_edge(edge_attr)
        
        for i in range(0, len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.bn_layers[i](x)
            

        graph_emb = self.pool(x, batch.long())

        graph_emb = self.conv_to_fc(self.conv_to_fc_softplus(graph_emb))
        graph_emb = self.conv_to_fc_softplus(graph_emb)
        

        return graph_emb

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            # # 用token_sim作为边属性，只保留那些实际连接的边的属性
            # edge_attr = token_sim[edge_index[0], edge_index[1]].unsqueeze(1)

            # 使用随机值作为边属性
            edge_attr = torch.rand(edge_index.size(1), 50)  # 假设边属性维度为1

            # 为每个token生成随机初始化的108维全局特征向量，并重复len(tokens)次
            glob_feat_single = torch.rand(108)  # 随机初始化一个108维的向量
            glob_feat = glob_feat_single.repeat(tokens.size(0), 1)  # 重复len(tokens)次

            pg_list.append(Data(x=tokens, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([i]).long(), glob_feat=glob_feat))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch




class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        # device = torch.device("cuda")
        # device = torch.device("cpu")

        # if torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        #     device = torch.device("cpu")

        device = graph_batch.x.device

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)
        pg.to(device)

        inner_edge_index = pg.edge_index
        inner_edge_attr = pg.edge_attr  # 获取提示图的边属性
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            g_edge_attr = g.edge_attr 
            
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            cross_edge_index.to(device)
            # # 计算交叉边的边属性，这里也使用相似度作为属性
            # cross_edge_attr = cross_sim[cross_edge_index[0], cross_edge_index[1] - token_num].unsqueeze(1)

            # 为交叉边使用随机初始化的属性
            cross_edge_attr = torch.rand(cross_edge_index.size(1), 50)  # 假设边属性维度为1
            cross_edge_attr = cross_edge_attr.to(device)

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)

            inner_edge_attr = inner_edge_attr.to(device)
            g_edge_attr = g_edge_attr.to(device)
            

            # print("inner_edge_attr device:", inner_edge_attr.device)
            # print("g_edge_attr device:", g_edge_attr.device)
            # print("cross_edge_attr device:", cross_edge_attr.device)

            edge_attr = torch.cat([inner_edge_attr, g_edge_attr, cross_edge_attr], dim=0)  # 合并边属性
            # edge_attr = g_edge_attr


            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch


class Prompt4deepGAT(LightPrompt):
    '''
    与heavyPrompt不同，这里的cross_edge的属性没有加入到edge_attr中
    '''
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(Prompt4deepGAT, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        :param graph_batch:
        :return:
        """
        device = graph_batch.x.device
        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)
        pg.to(device)

        inner_edge_index = pg.edge_index
        inner_edge_attr = pg.edge_attr  # 获取提示图的边属性
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            g_edge_attr = g.edge_attr 
            
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            cross_edge_index.to(device)
            # # 计算交叉边的边属性，这里也使用相似度作为属性
            # cross_edge_attr = cross_sim[cross_edge_index[0], cross_edge_index[1] - token_num].unsqueeze(1)

            # 为交叉边使用随机初始化的属性
            cross_edge_attr = torch.rand(cross_edge_index.size(1), 1)  # 假设边属性维度为1
            cross_edge_attr = cross_edge_attr.to(device)

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y
            # glob_feat_1 = g.glob_feat[0]
            # glob_feat = torch.repeat_interleave(glob_feat_1.unsqueeze(0), repeats=x.size(0), dim=0).to(device)
            glob_feat = torch.cat([pg.glob_feat, g.glob_feat], dim=0)

            # edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            edge_index = torch.cat([g_edge_index], dim=1)

            inner_edge_attr = inner_edge_attr.to(device)
            g_edge_attr = g_edge_attr.to(device)
            

            # print("inner_edge_attr device:", inner_edge_attr.device)
            # print("g_edge_attr device:", g_edge_attr.device)
            # print("cross_edge_attr device:", cross_edge_attr.device)

            # edge_attr = torch.cat([inner_edge_attr, g_edge_attr, cross_edge_attr], dim=0)  # 合并边属性
            edge_attr = torch.cat([g_edge_attr], dim=1)


            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, glob_feat=glob_feat)
            # print(f"data:{data}")
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch
class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre


@deprecated(version='1.0', reason="Pipeline is deprecated, use FrontAndHead instead")
class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'):
        warnings.warn("deprecated", DeprecationWarning)

        super().__init__()
        # load pre-trained GNN
        self.gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gcn_layer_num, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        self.gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in self.gnn.parameters():
            p.requires_grad = False

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch: Batch):
        prompted_graph = self.PG(graph_batch)
        graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre



if __name__ == '__main__':
    pass
