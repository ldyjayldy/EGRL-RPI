# EGRL-RPI

EGRL: Edge Generation-Guided Relation-Aware Learning for RNA-Protein Interaction Prediction.

## 工程结构

- `egrl/` : 公共代码（模型、图构建、训练、评估、冷启动、绘图）
- `experiments/` : 5 个实验入口，每个只有 main
- `data/` : 数据放置目录
- `results/` : 输出目录

## 安装

`pip install -r requirements.txt`

## 运行

**必须在 `experiments/` 下运行**，因为脚本用 `../data/...` 路径。

```bash
cd experiments
python standard_5fold_cv.py
python coldstart_molecule_holdout.py
python coldstart_cluster_holdout.py
python relation_negative_complexity.py
python soft_edge_quality.py
```

## 对应论文

| 脚本 | 论文 |
|---|---|
| standard_5fold_cv.py | 标准 5 折（RPI2241，Table 7/8） |
| coldstart_molecule_holdout.py | Molecule hold-out（Table 11 / Fig. 7） |
| coldstart_cluster_holdout.py | Cluster hold-out（Table 12 / Fig. 9） |
| relation_negative_complexity.py | 关系权重/负采样/复杂度（Table 4/5/6/9，RPI1807） |
| soft_edge_quality.py | Generator 排序质量（Table 2/3 / Fig. 4） |

## 与论文不一致处

见 `egrl/` 下各模块注释。主要几点：

- `ImplicitMetaPathLearner` 用 active node mask 计算 m_k，未严格按关系边节点集合。
- 多关系 GAT 传播用 `relation_logits` 原值，表 5 报告的是 `softmax(relation_logits)`。
- 生成器辅助损失 Eq.(9) 用总元素平均，而非 RNA/protein 两类分别平均。
- 冷启动测试负样本未实现 80%/40% identity 过滤。
- 数据集版本、编码维度需按实际 `data/` 内容核对。