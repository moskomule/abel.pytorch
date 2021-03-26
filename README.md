# abel.pytorch

PyTorch LR scheduler of ABEL from Lewkowycz 2021 "How to decay your learning rate".


## Requirements

* Python>=3.8
* PyTorch>=1.7.1

To run the example, you further need

* `homura` by `pip install -U homura-core==2020.12.0`
* `chika` by `pip install -U chika`

## Example

```commandline
python cifar10.py [--optim.name {abel,cosine,steps}] [--model {renst20, wrn28_2}] [--optim.gamma 0.1]
```

### Results: Test Accuracy (CIFAR-10)

Model       | ABEL| cosine |
---         | --- | ---    |
ResNet-20   | | 93.2   |
WRN28-2     | | 95.4   |
ResNeXT29   | | 95.8   |
