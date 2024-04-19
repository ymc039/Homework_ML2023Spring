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

# Neural Network Model

这部分我做了简单的修改，以便于后续调参

```python
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure in hyper-parameter: 'config', be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, config['layer'][0]),
            nn.ReLU(),
            nn.Linear(config['layer'][0], config['layer'][1]),
            nn.ReLU(),
            nn.Linear(config['layer'][1], 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```
