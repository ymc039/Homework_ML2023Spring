# Few-shot Classification

Omniglot datast

- Traing/validation set

	30 alphabets

	- mutiple characters in one alphabet
	- 20 images for one character

- Testing set

​	640 support and query pairs

# Baselines

- Simple(0.6)
- Medium(0.7)
- Strong(0.9)
- Boss(0.95)

# Results

## Simple(0.62329)

运行所给代码

## Medium(0.70681)

完成TODO

```python
# TODO: Finish the inner loop update rule
grads = torch.autograd.grad(loss, fast_weights.values())
fast_weights = OrderedDict((name, param - inner_lr*grad)
      for ((name, param), grad) in zip(fast_weights.items(), grads)
      )    
#raise NotImplementedError
```

```python
# TODO: Finish the outer loop update
meta_batch_loss.backward()
optimizer.step()
#raise NotimplementedError
```

## Strong(0.94488)

修改Meta Solver，同时训练120个epoch

```python
# TODO: Finish the inner loop update rule
grads = torch.autograd.grad(loss, fast_weights.values(), creat_graph=True)
fast_weights = OrderedDict((name, param - inner_lr*grad)
      for ((name, param), grad) in zip(fast_weights.items(), grads)
      )    
#raise NotImplementedError
```

## Boss(0.95395)

数据增强，训练150个epoch

```python
#MetaSolver函数中修改
for meta_batch in x:
    # Get data
    if torch.rand(1).item() > 0.5:
        times = 1 if torch.rand(1).item() > 0.5 else 3
        meta_batch = torch.rot90(meta_batch, times, [-1, -2])
```

