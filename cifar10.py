from functools import partial
from typing import List, Tuple

import chika
import homura
import torch
import torch.nn.functional as F
from homura import lr_scheduler, reporters, trainers
from homura.vision import DATASET_REGISTRY, MODEL_REGISTRY

from abel import ABEL as _ABEL


def ABEL(gamma):
    return partial(_ABEL, gamma=gamma)


@chika.config
class Optim:
    epochs: int = 200
    name: str = chika.choices("abel", "steps", "cosine")
    lr: float = 0.1
    weight_decay: float = 5e-4
    gamma: float = 0.1
    steps: List[int] = chika.sequence(100, 150)


@chika.config
class Config:
    optim: Optim
    model: str = chika.choices("resnet20", "resnet56", "se_resnet56", "wrn28_2", "resnext29_32x4d")
    batch_size: int = 128
    use_amp: bool = False
    seed: int = 1
    gpu: int = chika.bounded(0, 0, torch.cuda.device_count())


class Trainer(trainers.SupervisedTrainer):
    def iteration(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        if not self.is_train:
            return super().iteration(data)

        input, target = data

        def closure():
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_f(output, target)
            loss.backward()
            return loss

        loss = self.optimizer.step(closure)
        self.reporter.add("loss", loss)


def _main(cfg):
    model = MODEL_REGISTRY(cfg.model)(num_classes=10)
    train_loader, test_loader = DATASET_REGISTRY("cifar10")(cfg.batch_size, num_workers=4, download=True)
    optimizer = homura.optim.SGD(lr=cfg.optim.lr, momentum=0.9, weight_decay=cfg.optim.weight_decay)
    scheduler = {
        "cosine": lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, 4, 5),
        "abel": ABEL(cfg.optim.gamma),
        "steps": lr_scheduler.MultiStepLR(milestones=cfg.optim.steps, gamma=cfg.optim.gamma),
    }[cfg.optim.name]

    with Trainer(
        model,
        optimizer,
        F.cross_entropy,
        reporters=[reporters.TensorboardReporter(".")],
        scheduler=scheduler,
        use_amp=cfg.use_amp,
    ) as trainer:
        for _ in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


@chika.main(cfg_cls=Config, strict=True)
def main(cfg: Config):
    torch.cuda.set_device(cfg.gpu)
    with homura.set_seed(cfg.seed):
        _main(cfg)


if __name__ == "__main__":
    main()
