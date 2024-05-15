# 数据处理和有监督模型
参考论文：
1. Benchmarking graph neural networks for materials chemistry-2021-nature
2. Crystal twins: self-supervised learning for crystalline material property prediction-2022-npj
3. Calable Deeper Graph Neural Networks for High-performance Materials Property Prediction-2022-patterns

## 数据处理
usage：
```python
'''
dictionary_blank.json
dictionary_default.json
process.py会用到
'''
# 首先需要加载数据集，并存储成json格式，晶体结构存储于json文件中，预测目标存储于target.csv中
python laod_data.py
# 对数据集进行处理，方便模型进行读取，处理好的文件存储于./data/{dataset_name}/processed/data.pt中
python process.py
```



