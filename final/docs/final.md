# 人工智能安全大作业 -- 对抗攻防

## 简介
本次实验利用`PyTorch`复现了`Towards Deep Learning Models Resistant to Adversarial Attacks`文章中提到的白盒攻击方法，在`MNIST`数据集上进行了测试，证明了利用`PGD`生成对抗样本的可行性，并探究了`PGD`攻击时的参数与攻击效果的关系。

论文中提到的的攻击代码: https://github.com/MadryLab/mnist_challenge (使用`TensorFlow`实现)

## 算法描述

目标描述

![image-20220614141042008](/Users/krrrr/Library/Application%20Support/typora-user-images/image-20220614141042008.png)

*θ*：模型参数

L：损失函数

*δ*：扰动

S：允许的扰动范围

寻找模型参数θ最小化期待的最大损失



攻击——内部最大化损失问题 

防御——外部最小化期待问题

我们的任务为生成对抗攻击样本，采用**projected gradient descent (PGD)**的多步方法

![image-20220614141639566](/Users/krrrr/Library/Application%20Support/typora-user-images/image-20220614141639566.png)

代码实现：

```python
loss, num_correct, accuracy = self.model(x, y)
grad = torch.autograd.grad(loss, x)[0]
x1 = self.a * torch.sign(grad) + x
```

## 代码细节

### 文件结构

```
.
├── README.md
├── checkpoints
├── config.json
├── eval.py
├── eval_logs
├── eval_pl.py
├── lightning_logs
├── mnist
├── model_bare.py
├── model_pl.py
├── optuna_pl.py
├── pgd_attack.py
├── pgd_attack_pl.py
├── requirements.txt
├── train_bare.py
├── train_pl.py
└── utils.py
```

> train_pl.py 正常训练模型
>
> model_pl.py 模型
>
> pgd_attack.py 生成对抗样本
>
> eval.py 评估对抗样本对正常训练模型进行白盒攻击的成功率
>
> config.json 配置文件
> 
> utils.py 初始化数据集 & 设置checkpoint

### 训练过程

我们进行了`10`个epoch的训练，由于模型规模较小，可以通过loss和acc的变化确定模型已经收敛



此外，基于利用pytorch-lightning包装的模型，我们使用lightning提供的`trainer`进行重构，以便在多卡集群上进行训练和验证。


```python
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
```

### 模型结构
整体的模型结构复现了论文中的MNIST数据集图片分类模型结构。

### PGD攻击
#### `class LinfPGDAttack`
定义了一个 `LinfPGDAttack` 类，函数 `perturb` 使用前面所提到的算法产生对抗样本。

可以在配置文件的`random_start`参数中设置是否给原始样本添加任意扰动后，再开始进行梯度下降。

#### `__main__`
默认情况下，在main函数中，如果直接运行`pgd_attack.py`文件，会利用定义好的PGD类，基于`MNIST`提供的test数据集以生成对抗样本，并将对抗样本保存到numpy的输出`.npy`文件中。


### eval.py

正常样本 vs 对抗样本

```python
loss_nat, num_correct_nat, accuracy_nat = model(x_batch, y_batch)
loss_adv, num_correct_adv, accuracy_adv = model(x_batch_adv, y_batch_adv)
```

```python
avg_xent_nat = total_xent_nat / num_eval_examples
avg_xent_adv = total_xent_adv / num_eval_examples

acc_nat = total_corr_nat / num_eval_examples
acc_adv = total_corr_adv / num_eval_examples
```

## 攻击效果

to be continued...