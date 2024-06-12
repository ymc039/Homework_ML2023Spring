# Anomaly Detection

- Train: 
	- 100000 human faces
- Test: 
	- About 10000 from the same distribution with training data (label 0)
	- About 10000 from another distribution (anomalies, label 1)

# Baselines

- Simple(0.53458)
- Medium(0.71931)
- Strong(0.78950)
- Boss(0.82221)

# Results

## Simple(0.72827)

运行代码

样例代码给了fcn、cnn和vae三种形式，运行得分为0.72827、0.68283和0.53477。

## Medium(0.75595)

更改fcn的结构，其余不变

```python
self.encoder = nn.Sequential(
    nn.Linear(64 * 64 * 3, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(), 
    nn.Linear(256, 64), 
    nn.ReLU(), 
    nn.Linear(64, 10)
)

self.decoder = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(), 
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 1024),
    nn.ReLU(), 
    nn.Linear(1024, 64 * 64 * 3), 
    nn.Tanh()
)
```

## Strong(0.79203)

首先根据提示使用了Multi-encoder的方法，拼接了fcn和cnn，同时调整`batchsize=128`和`epoch=100`，得分为0.74909，还不如fcn。。。

尝试了ResNet作为encoder，四层网络，每层残差块数量为[2, 1, 1, 1]，具体实现参考了[yaoweizhang]([LHY2022-SPRING/Hw08/answer/2022-hw8-strong-resnet.ipynb at main · yaoweizhang/LHY2022-SPRING (github.com)](https://github.com/yaoweizhang/LHY2022-SPRING/blob/main/Hw08/answer/2022-hw8-strong-resnet.ipynb))，得分0.77027。

在[这篇博客]([李宏毅2022机器学习HW8解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/529030465))提到了可以增加一个辅助网络提高模型的表现，辅助网络的结构与原始模型中的decoder一致，在训练过程中与模型一同训练，只有损失函数形式不同。模型损失函数为MSE，辅助网络损失函数为
$$L_{a} = \exp\left(\frac{\text{criterion}(output_{a}, output)}{temperature}\right) \cdot \text{criterion}(output_{a}, img)$$

`temperature`随着训练过程，每两个epoch增加1

仍然使用ResNet作为encoder

decoder结构为

```python
self.decoder = nn.Sequential(
nn.Linear(64, 64*4*4),
nn.BatchNorm1d(64*4*4),
nn.ReLU(),
nn.Unflatten(1, (64, 4, 4)),
nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
nn.BatchNorm2d(128),
nn.ReLU(),
nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
nn.BatchNorm2d(128),
nn.ReLU(),
nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
nn.BatchNorm2d(128),
nn.ReLU(),
nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
nn.Tanh(),
)
```

训练150个epoch，得分0.79203。

在训练过程中，decoder层的梯度变化如图

![](grad_comparision.png)

博客作者没有提到这样做的原因，但这篇文章（[Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning]([arxiv.org/pdf/2012.09816](https://arxiv.org/pdf/2012.09816))）或许可以作出一定解释。这是一种在线的自蒸馏技巧（我猜的:p）

## Boss(0.84040)

主要参考了[ml-spring2023/HW08/hw8.ipynb at main · shenjiekoh/ml-spring2023 (github.com)](https://github.com/shenjiekoh/ml-spring2023/blob/main/HW08/hw8.ipynb)，选择了fcn结构。

```python
self.encoder = nn.Sequential(
    nn.Linear(64 * 64 * 3, 4096),
    nn.ReLU(),
    nn.Linear(4096, 2048),
    nn.ReLU(), 
    nn.Linear(2048, 1024), 
    nn.ReLU(), 
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)    # Hint: dimension of latent space can be adjusted

self.decoder = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 4096),
    nn.ReLU(),
    nn.Linear(4096, 64 * 64 * 3), 
    nn.Tanh()
)
```

参考[ML2022/HW08/ML2022Spring_HW8.ipynb at master · ncku-yee/ML2022 (github.com)](https://github.com/ncku-yee/ML2022/blob/master/HW08/ML2022Spring_HW8.ipynb)，在数据预处理中增加

```python
self.transform = transforms.Compose([
  transforms.Lambda(lambda x: x.to(torch.float32)),
  transforms.Resize((256, 256)),
  transforms.CenterCrop((196, 196)),
  transforms.Resize((64, 64)),
  transforms.Lambda(lambda x: 2. * x/255. - 1.),
])
```

得分0.84040。使用自蒸馏后，得分0.83440，略有下降。

[NTU_ML_2023_Spring/hw8/R11921091_ML2023Spring_HW8_Bonus_Report.pdf at main · ianyang66/NTU_ML_2023_Spring (github.com)](https://github.com/ianyang66/NTU_ML_2023_Spring/blob/main/hw8/R11921091_ML2023Spring_HW8_Bonus_Report.pdf)提到可以提升得分至0.87+，但我没试。
