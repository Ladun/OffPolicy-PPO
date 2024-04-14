
import torch

class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):

        if max_steps < warmup_steps:
            max_steps = warmup_steps + 1;

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return max(1.0 - float((step - warmup_steps) / (max_steps - warmup_steps)), 0)

        super(WarmupLinearSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)