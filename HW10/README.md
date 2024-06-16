# Adversarial Attack

- Images:
	- (32 * 32 RGB images) * 200
	- 10 classes (airplane, automobile, bird, …)
	- 20 images for each class

## Baselines

- Simple(acc<=0.70)
- Medium(acc<=0.50)
- Strong(acc<=0.25)
- Boss(acc<=0.10)

## Results

### Simple(0.59000)

运行所给代码即可

### Strong(0.01000)

根据提示，使用ifgsm，效果出人意料地好！

```python
# alpha and num_iter can be decided by yourself
alpha = 0.8/255/std

def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x.detach().clone()
    ################ TODO: Medium baseline #######################
    # write a loop with num_iter times
    for i in range(num_iter):
        # TODO: Each iteration, execute fgsm
        x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)
#         print(epsilon)
#         print(x_adv[1, :, :].size())
        # x_adv和x的无穷范数距离不能超过epsilon
        for channel in range(3):
            x_adv[channel, :, :] = torch.clip(x_adv[channel, :, :], x[channel, :, :] - epsilon, x[channel, :, :] + epsilon)
```

### Boss(0.00000)

使用DIM-MIFGSM，同时使用EnsembleModel。

```python
def dim_mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=50, decay=1.0, p=0.5):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv_raw = x_adv.clone()
        if torch.rand(1).item() >= p:
            #resize img to rnd X rnd
            rnd = torch.randint(29, 33, (1,)).item()
            x_adv = transforms.Resize((rnd, rnd))(x_adv)
            #padding img to 32 X 32 with 0
            left = torch.randint(0, 32 - rnd + 1, (1,)).item()
            top = torch.randint(0, 32 - rnd + 1, (1,)).item()
            right = 32 - rnd - left
            bottom = 32 - rnd - top
            x_adv = transforms.Pad([left, top, right, bottom])(x_adv)
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # TODO: Momentum calculation
        # grad = .....   
        grad = x_adv.grad.detach()
        grad = decay * momentum + grad/(grad.abs().sum() + 1e-8)
        momentum = grad
        x_adv = x_adv_raw + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv
```

```python
class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
        
    def forward(self, x):
        ensemble_logits = None
        #################### TODO: boss baseline ###################
        for i, m in enumerate(self.models):
            ensemble_logits = m(x) if i == 0 else ensemble_logits + m(x)
        # TODO: sum up logits from multiple models  
        return ensemble_logits/len(self.models)
...
model_names = [
    'nin_cifar10',
    'resnet20_cifar10',
    'preresnet20_cifar10'
]

for model_name in model_names:
    model_checker(model_name)

ensemble_model = ensembleNet(model_names).to(device)
ensemble_model.eval()
```

