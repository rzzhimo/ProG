from ProG.utils import seed_everything, seed

seed_everything(seed)

from ProG import PreTrain
from ProG.utils import mkdir, load_data4pretrain
from ProG.prompt import GNN, LightPrompt, HeavyPrompt, CGCNN, Prompt4deepGAT
from ProG.deep_gatgnn import DEEP_GATGNN
from torch import nn, optim
from ProG.data import multi_class_NIG, regression_matbench, binary_class_matbench
import torch
from torch_geometric.loader import DataLoader
from ProG.eva import acc_f1_over_batches, regression_metrics_over_batches


# this file can not move in ProG.utils.py because it will cause self-loop import
def model_create(dataname, pretext, gnn_type, num_class, task_type='multi_class_classification', tune_answer=False):
    if task_type in ['multi_class_classification', 'regression']:
        input_dim, hid_dim = 114, 64 # 这个hid_dim要与预训练所用的维度数相匹配
        lr, wd = 0.001, 0.00001
        tnpc = 100  # token number per class

        # load pre-trained GNN
        if gnn_type == "CGCNN":
            gnn = CGCNN(input_node_dim=input_dim, input_edge_dim=50, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, pool=None,
                             gnn_type=gnn_type)
        elif gnn_type == "deeperGAT":
            gnn = DEEP_GATGNN(num_features=input_dim, dim1=100, dim2=100, num_edge_features=50, output_dim=100, gc_count=5)
        else:
            gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)

        pre_train_path = './pre_trained_gnn/{}.{}.{}.pth'.format(dataname, pretext, gnn_type)
        # pre_train_path = "./pre_trained_gnn/matbench_dielectric.SimGRACE.TransformerConv.pth"
        # pre_train_path = "./pre_trained_gnn/matbench_mp_is_metal.GraphCL.CGCNN.pth"
        # pre_train_path = "./pre_trained_gnn/all_in_one.GraphCL.deeperGAT.pth"

        # gnn.load_state_dict(torch.load(pre_train_path))
        gnn.load_state_dict(torch.load(pre_train_path, map_location=torch.device(device)))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in gnn.parameters():
            p.requires_grad = False

        if tune_answer:
            if gnn_type == "CGCNN":
                # 加上提示节点之间的边与提示节点和原图节点之间的边
                PG = HeavyPrompt(token_dim=input_dim, token_num=1, cross_prune=0.1, inner_prune=0.3)
                # PG = Prompt4deepGAT(token_dim=input_dim, token_num=5, cross_prune=0.1, inner_prune=0.3)
            elif gnn_type == "deeperGAT":
                PG = Prompt4deepGAT(token_dim=input_dim, token_num=5, cross_prune=0.1, inner_prune=0.3)
            else:
                PG = HeavyPrompt(token_dim=input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3)
        else:
            PG = LightPrompt(token_dim=input_dim, token_num_per_group=tnpc, group_num=num_class, inner_prune=0.01)

        opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                         lr=lr,
                         weight_decay=wd)

        if task_type == 'regression':
            # lossfn = nn.MSELoss(reduction='mean')
            lossfn = nn.L1Loss()
        else:
            # lossfn = nn.CrossEntropyLoss(reduction='mean')
            lossfn = nn.NLLLoss()

        if tune_answer:
            # if task_type == 'regression' and 'matbench' in dataname:
            #     answering = torch.nn.Sequential(
            #         torch.nn.Softplus(),
            #         torch.nn.Linear(hid_dim, 1))
            if task_type == 'regression' and 'matbench' in dataname:
                answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, hid_dim),
                # torch.nn.Dropout(0.5),  # 添加Dropout层，其中0.5是dropout比例
                torch.nn.LayerNorm(hid_dim),  # 添加LayerNorm层
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hid_dim, 1)
            )
            elif task_type == 'regression':
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Sigmoid())
            else:
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Softmax(dim=1))

            opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=0.01,
                                    weight_decay=0.00001)
        else:
            answering, opi_answer = None, None
        gnn.to(device)
        PG.to(device)
        return gnn, PG, opi, lossfn, answering, opi_answer
    else:
        raise ValueError("model_create function hasn't supported {} task".format(task_type))


def pretrain():
    mkdir('./pre_trained_gnn/')

    pretext = 'GraphCL'  # 'GraphCL', 'SimGRACE'
    gnn_type = 'GCN'  # 'GAT', 'GCN', 'TransformerConv'
    dataname, num_parts, batch_size = 'CiteSeer', 200, 10

    print("load data...")
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

    print("create PreTrain instance...")
    pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)

    print("pre-training...")
    pt.train(dataname, graph_list, batch_size=batch_size,
             aug1='dropN', aug2="permE", aug_ratio=None,
             lr=0.01, decay=0.0001, epochs=100)


def prompt_w_o_h(dataname="CiteSeer", pretext="GraphCL", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification'):
    _, _, train_list, test_list = multi_class_NIG(dataname, num_class, shots=100)

    train_loader = DataLoader(train_list, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=100, shuffle=True)

    gnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(dataname, pretext, gnn_type, num_class, task_type, False)
    # Here we have: answering, opi_answer=None, None
    lossfn.to(device)
    

    prompt_epoch = 200
    # training stage
    PG.train()
    for j in range(1, prompt_epoch + 1):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):
            # print(train_batch)
            train_batch = train_batch.to(device)
            if gnn.gnn_type == "CGCNN":
                emb0 = gnn(train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.batch)
                pg_batch = PG.inner_structure_update()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.edge_attr, pg_batch.batch)
            else:
                emb0 = gnn(train_batch.x, train_batch.edge_index, train_batch.batch)
                pg_batch = PG.inner_structure_update()
                pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            # cross link between prompt and input graphs
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            if task_type == 'multi_class_classification':
                sim = torch.softmax(dot, dim=1)
            elif task_type == 'regression':
                sim = torch.sigmoid(dot)  # 0-1
            else:
                raise KeyError("task type error!")

            train_loss = lossfn(sim, train_batch.y)
            opi_pg.zero_grad()
            train_loss.backward()
            opi_pg.step()
            running_loss += train_loss.item()

            if batch_id % 5 == 4:  # report every 5 updates
                last_loss = running_loss / 5  # loss per batch
                print(
                    'epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, prompt_epoch, batch_id+1, len(train_loader),
                                                                      last_loss))

                running_loss = 0.

        if j % 5 == 0:
            PG.eval()
            PG = PG.to("cpu")
            gnn = gnn.to("cpu")
            acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device)

            PG.train()
            PG = PG.to(device)
            gnn = gnn.to(device) 


def train_one_outer_epoch(epoch, train_loader, opi, lossfn, gnn, PG, answering, task_type):
    for j in range(1, epoch + 1):
        running_loss = 0.
        # bar2=tqdm(enumerate(train_loader))
        for batch_id, train_batch in enumerate(train_loader):  # bar2
            # print(train_batch)
            train_batch = train_batch.to(device)
            prompted_graph = PG(train_batch)
            # print(prompted_graph)
            if gnn.gnn_type == "CGCNN":
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.edge_attr,prompted_graph.batch)
            elif gnn.gnn_type == "deeperGAT":
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.edge_attr, prompted_graph.glob_feat, prompted_graph.batch)
            else:
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            # print(graph_emb)
            pre = answering(graph_emb)
            if task_type == "regression":
                # train_batch.y = train_batch.y.unsqueeze(dim=1)  # 这将改变 train_batch.y 的形状为 [n, 1]，其中 n 是 batch_size
                train_batch.y = train_batch.y
                # 对于回归任务，通常保持train_batch.y为浮点类型以匹配预测值的类型
            else:
                # 对于分类任务，确保train_batch.y是long类型
                train_batch.y = train_batch.y.long()  # 修改此行以转换类型
            # print(f"train_batch.y:{train_batch.y}")
            
            train_loss = lossfn(pre, train_batch.y)
            # print('\t\t==> answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
            #                                                                     train_loss.item()))

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

            if batch_id % 5 == 4:  # report every 5 updates
                last_loss = running_loss / 5  # loss per batch
                # bar2.set_description('answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
                #                                                                     last_loss))
                print(
                    'epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, epoch, batch_id, len(train_loader), last_loss))

                running_loss = 0.


def prompt_w_h(dataname="CiteSeer", pretext="GraphCL" ,gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification'):
    _, _, train_list, test_list = multi_class_NIG(dataname, num_class, shots=100)

    # 看看train_list, test_list的数据
    print(f"train_list[0:5]:{train_list[0:5]}")
    print(f"test_list[0:5]:{test_list[0:5]}")
    
    # 输出图的个数
    print(f"train_list图的个数:{len(train_list)}")
    print(f"test_list图的个数:{len(test_list)}")

    train_loader = DataLoader(train_list, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=10, shuffle=True)

    # 看看train_loader, test_loader的数据
    print(f"train_loader:{train_loader}",train_loader)
    print(f"test_loader:{test_loader}",test_loader)

    gnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(dataname, pretext, gnn_type, num_class, task_type, True)
    answering.to(device)

    # inspired by: Hou Y et al. MetaPrompting: Learning to Learn Better Prompts. COLING 2022
    # if we tune the answering function, we update answering and prompt alternately.
    # ignore the outer_epoch if you do not wish to tune any use any answering function
    # (such as a hand-crafted answering template as prompt_w_o_h)
    outer_epoch = 10
    answer_epoch = 1  # 50
    prompt_epoch = 1  # 50

    # training stage
    for i in range(1, outer_epoch + 1):
        print(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
        # tune task head
        answering.train()
        PG.eval()
        train_one_outer_epoch(answer_epoch, train_loader, opi_answer, lossfn, gnn, PG, answering, task_type)

        print("{}/{}  frozen gnn | *tune prompt |frozen answering function...".format(i, outer_epoch))
        # tune prompt
        answering.eval()
        PG.train()
        train_one_outer_epoch(prompt_epoch, train_loader, opi_pg, lossfn, gnn, PG, answering, task_type)

        # testing stage
        answering.eval()
        PG.eval()
        acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device = device)     

def     prompt_w_h_mb(dataname="matbench_dielectric", pretext="GraphCL", gnn_type="GCN", num_class=1, task_type="regression"):
    if task_type == "regression":
        _, _, train_list, test_list = regression_matbench(dataname, shots=50, train_rate=0.6)
    else:
        _, _, train_list, test_list = binary_class_matbench(dataname, shots=50, train_rate=0)

    # 看看train_list, test_list的数据
    print(f"train_list[0:5]:{train_list[0:5]}")
    print(f"test_list[0:5]:{test_list[0:5]}")
    # 看看train_list前5个y值
    for i in range(5):
        print(f"train_list[{i}].y:{train_list[i].y}")
    # 输出图的个数
    print(f"train_list图的个数:{len(train_list)}")
    print(f"test_list图的个数:{len(test_list)}")

    train_loader = DataLoader(train_list, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=100, shuffle=True)
    
    # 看看train_loader, test_loader的数据
    print(f"train_loader:{train_loader}")
    print(f"test_loader:{test_loader}")

    gnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(dataname, pretext, gnn_type, num_class, task_type, True)
    answering.to(device)

    # inspired by: Hou Y et al. MetaPrompting: Learning to Learn Better Prompts. COLING 2022
    # if we tune the answering function, we update answering and prompt alternately.
    # ignore the outer_epoch if you do not wish to tune any use any answering function
    # (such as a hand-crafted answering template as prompt_w_o_h)
    outer_epoch = 20
    answer_epoch = 10  # 50
    prompt_epoch = 10  # 50

    # training stage
    for i in range(1, outer_epoch + 1):
        print(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
        # tune task head
        answering.train()
        PG.eval()
        train_one_outer_epoch(answer_epoch, train_loader, opi_answer, lossfn, gnn, PG, answering, task_type)

        print("{}/{}  frozen gnn | *tune prompt |frozen answering function...".format(i, outer_epoch))
        # tune prompt
        answering.eval()
        PG.train()
        train_one_outer_epoch(prompt_epoch, train_loader, opi_pg, lossfn, gnn, PG, answering, task_type)

        # testing stage
        answering.eval()
        PG.eval()
        if task_type == "regression":
            regression_metrics_over_batches(test_loader, PG, gnn, answering, device = device)
        else:
            acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device = device)
    # final testing stage
    answering.eval()
    PG.eval()
    if task_type == "regression":
        regression_metrics_over_batches(test_loader, PG, gnn, answering, device = device)
    else:
        acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device = device)
if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda:0")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    
    print(device)
    # device = torch.device('cpu')
    # deeperGAT,CGCNN, TransformerConv, GCN, GAT
    # GraphCL,SimGRACE
    # pretrain()
    # prompt_w_o_h(dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification')
    # prompt_w_h(dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification')
     # 'matbench_mp_e_form', 'matbench_mp_is_metal', 'matbench_mp_gap', 'matbench_jdft2d', 'all_in_one', 'matbench_log_kvrh', 'matbench_dielectric'
    # prompt_w_h_mb(dataname="matbench_jdft2d", pretext="SimGRACE", gnn_type="TransformerConv", num_class=1, task_type='regression')
    prompt_w_h_mb(dataname="matbench_mp_is_metal",pretext="GraphCL", gnn_type="CGCNN", num_class=2, task_type='multi_class_classification')