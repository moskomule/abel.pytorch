import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ABEL(_LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1, debug: bool=False) -> None:
        self.optimizer = optimizer
        self.gamma = gamma
        self.debug = debug
        self.norms = []
        self.reached_minimum = False

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        norm = sum([sum([p.view(-1).norm(2) for p in group["params"]]) for group in self.optimizer.param_groups])
        self.norms.append(norm.cpu().item())
        gamma = 1
        if len(self.norms) > 3 and (self.norms[-1] - self.norms[-2]) * (self.norms[-2] - self.norms[-3]) < 0:
            if self.reached_minimum:
                self.reached_minimum = False
                gamma = self.gamma
                if self.debug:
                    print(f"ABEL decreases LR at {self.last_epoch}th step")
            else:
                self.reached_minimum = True
        return [group["lr"] * gamma for group in self.optimizer.param_groups]
