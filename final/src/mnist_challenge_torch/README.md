
## How To Run?
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
The `MNIST` dataset will be downloaded automatically.
```bash
python train_pl.py # run with pytorch-lightning
python train_bare.py # run without pytorch-lightning
```

### 3. Run Attack
The PGD attacking method is implemented to generate adversarial examples.
```bash
python pgd_attack_pl.py # run with pytorch-lightning
python pgd_attack.py # run without pytorch-lightning
```

## 4. Run Evaluation
Evaluation result is printed in the console with the natural accuracy and the adversarial accuracy.
```bash
python eval_pl.py # run with pytorch-lightning
python eval.py # run without pytorch-lightning
```

## 5. Visualization
The `TensorBoard` visualization is implemented to visualize the training process.

Logs are saved to `./lightning_logs/version_xx`.

In order to run `TensorBoard` server, set the `--logdir` flag to the directory of logs. For example,
```bash
tensorboard --logdir=./lightning_logs
```


