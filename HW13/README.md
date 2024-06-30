# Network Compression

- Food dataset with 11 classes
- Train: 9993 labeled imgs
- Valid: 4432 labeled imgs
- Evaluation: 2218 imgs

## Baselines

- Simple(0.46528)
- Medium(0.69161)
- Strong(0.78629)
- Boss(0.82146)

# Results

## Simple(0.45807)

运行所给代码

将`dataloader`中的`num_workers`设置为4，可以加快训练。如果因此出现了多线程错误，将`from tqdm.auto import tqdm`改为`from tqdm import tqdm`即可。

## Medium(0.75202)

自定义损失函数代码如下

```python
CE = nn.CrossEntropyLoss()
def loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.5, temperature=20.0):
    # ------------TODO-------------
    # Refer to the above formula and finish the loss function for knowkedge distillation using KL divergence loss and CE loss.
    # If you have no idea, please take a look at the provided useful link above.
    loss_ce = F.cross_entropy(student_logits, labels)
    p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    loss_kl = F.kl_div(p, q, reduction='batchmean', log_target=False)
    loss = alpha * temperature * temperature * loss_kl + (1 - alpha) * loss_ce
    return loss
```

训练700个epoch

## Boss(0.82687)

修改`train_transform`

```python
train_tfm = transforms.Compose([
    # add some useful transform or augmentation here, according to your experience in HW3.
#     transforms.Resize(256),  # You can change this
#     transforms.CenterCrop(224), # You can change this, but be aware of that the given teacher model's input size is 224.
#     # The training input size of the provided teacher model is (3, 224, 224).
#     # Thus, Input size other then 224 might hurt the performance. please be careful.
#     transforms.RandomHorizontalFlip(), # You can change this.
    transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(180),
    transforms.RandomAffine(30),
    transforms.ToTensor(),
    normalize,
])
```

Depthwise与Pointwise卷积实现参考[机器学习手艺人](Depthwise卷积与Pointwise卷积)

```python
def dwpw_conv(ic, oc, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(ic, ic, kernel_size, stride=stride, padding=padding, groups=ic), #depthwise convolution
        nn.BatchNorm2d(ic),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(ic, oc, 1), # pointwise convolution
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(0.01, inplace=True)
    )
```

由此修改student模型结构

另外，训练中损失函数要逐层与teacher模型对齐，有点类似深度监督的感觉

config如下

```python
cfg = {
    'dataset_root': '/kaggle/input/ml2023spring-hw13/Food-11',
    'save_dir': '/kaggle/working/',
    'exp_name': "simple_baseline",
    'batch_size': 128,
    'lr': 1e-3,
    'seed': 20220013,
    'loss_fn_type': 'KD', # simple baseline: CE, medium baseline: KD. See the Knowledge_Distillation part for more information.
    'weight_decay': 1e-5,
    'grad_norm_max': 10,
    'n_epochs': 600, # train more steps to pass the medium baseline.
    'patience': 60,
}
```

由于kaggle的时间限制和作者没有修改dataloader参数的原因，只训练了280个epoch。我加速了训练后，600个epoch只需要7个小时。另外，训练过程中，速度瓶颈并不在GPU而在CPU，仍有优化空间，不过本人水平有限，就留待有缘人来解决了。