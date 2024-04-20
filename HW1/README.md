# 数据解析

- covid_train.txt: 训练数据
- covid_test.txt: 测试数据

数据大体分为三个部分：id, states: 病例对应的地区, 以及其他数据
- id: sample 对应的序号。
- states: 对 sample 来说该项为 one-hot vector。从整个数据集上来看，每个地区的 sample 数量是均匀的，可以使用`pd.read_csv('./covid_train.csv').iloc[:,1:34].sum()`来查看，地区 sample 数量为 88/89。
- 其他数据: 这一部分最终应用在助教所给的 sample code 中的 select_feat。

    - Covid-like illness (5) 新冠症状

      - cli, ili ...

    - Behavier indicators (5) 行为表现

      - wearing_mask、travel_outside_state ... 是否戴口罩，出去旅游 ...

    - Belief indicators (2) 是否相信某种行为对防疫有效

      - belief_mask_effective, belief_distancing_effective. 相信戴口罩有效，相信保持距离有效。

    - Mental indicator (2) 心理表现

      - worried_catch_covid, worried_finance.  担心得到covid，担心经济状况

    - Environmental indicators (3) 环境表现

      - other_masked_public, other_distanced_public ... 周围的人是否大部分戴口罩，周围的人是否大部分保持距离 ...

    - Tested Positive Cases (1) 检测阳性病例，该项为模型的预测目标

      - **tested_positive (this is what we want to predict)** 单位为百分比，指有多少比例的人  

# Baseline

- Simple Baseline (1.96993)
	- 运行所给的 sample code。
- Medium Baseline (1.15678)
	- 特征选择，简单手动的选择你认为关联性较大的特征。
- Strong Baseline (0.92619)
	- 尝试不同的优化器（如：Adam）。
	- 应用 L2 正则化（SGD/Adam ... 优化器参数中的 weight_decay)
- Boss Baseline (0.81456)
	- 尝试更好的特征选择，可以使用 sklearn.feature_selection.SelectKBest。
	- 尝试不同的模型架构（调整 my_module.layers）
	- 调整其他超参数

# 结果

- sample(2.00579/1.89852)
- remove_states(0.91898/0.88883)
- remove_states+weight_decay(0.91905/0.88879)
- remove_states+weight_decay+10000epochs(0.91323/0.88164)
- 8feats(0.92196/0.89200)
- resNN*3(0.95993/0.91598)
