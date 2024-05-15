import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random

from ProG.prompt import GNN, CGCNN
from ProG.utils import gen_ran_output,load_data4pretrain,mkdir, graph_views, load_mbdata4pretrain
from ProG.pre_train import PreTrain

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)
    # device = torch.device('cpu')

    mkdir('./pre_trained_gnn/')
    # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
    # only selected pre-trained models will be moved into (1) so that we can keep reproduction

    pretext = 'GraphCL' 
    # pretext = 'SimGRACE' 
    # gnn_type = 'TransformerConv'  
    # gnn_type = 'GAT'
    # gnn_type = 'CGCNN'
    gnn_type = 'deeperGAT'
    # gnn_type = 'TransformerConv'

    # dataname, num_parts = 'CiteSeer', 200
    # graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)
    dataname = 'matbench_mp_e_form' # 'matbench_log_gvrh','matbench_mp_e_form', 'matbench_mp_is_metal', 'matbench_perovskites','matbench_mp_gap', 'matbench_jdft2d', 'all_in_one', 'matbench_log_kvrh', 'matbench_dielectric'
    graph_list, input_dim, hid_dim = load_mbdata4pretrain(dataname)

    pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)
    pt.model.to(device) 
    pt.train(dataname, graph_list, batch_size=100, aug1='dropN', aug2="permE", aug_ratio=0.2, lr=0.01, decay=0.0001,epochs=200)
