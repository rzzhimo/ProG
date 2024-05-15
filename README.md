# Crystal Prompt

## 项目结构
```
.
|—— Dataset     所用的数据集    
    |—— preprocess_data         处理数据所用的代码
    |—— matbench_*              matbench的所有数据集
    |—— all_in_one              为所有数据集合并为一起的大数据集，主要用于Pre-training
|—— pre_trained_gnn             不同的Pre-training strategies预训练跑出来的GNN参数
|—— ProG                        主要部件
    |—— deep_gatgnn.py          deep_gatgnn 作为GNN Encoder
|—— pretrain_matbench.py        预训练入口
|—— no_meta_demo.py             prompt tuning 入口

```

## 用法
1. usage
```python
'''
跑预训练代码
可调节参数：
pretext = 'GraphCL' 
# pretext = 'SimGRACE' 
gnn_type = 'CGCNN'
# gnn_type = 'deeperGAT'
dataname = 'matbench_dielectric'
'''
python pretrain_matbench.py

'''
跑prompt-tuning代码
可调节参数：
pretext = 'GraphCL' 
# pretext = 'SimGRACE' 
gnn_type = 'CGCNN'
# gnn_type = 'deeperGAT'
dataname = 'matbench_dielectric'
'''
python no_meta_demo.py
```