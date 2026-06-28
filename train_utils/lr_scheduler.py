from diffusers.optimization import get_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup_and_decay(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    constant_steps: int = -1,
    eps: float = 1e-5,
) -> LambdaLR:
    """Warmup → constant → linear-decay LR schedule.

    If `constant_steps <= 0`, the constant region extends from warmup end to training end
    (i.e. no decay tail).
    """
    if constant_steps <= 0:
        constant_steps = num_training_steps - num_warmup_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + constant_steps:
            return 1.0
        return max(
            eps,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps - constant_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_lr_scheduler(args, optimizer, step_multiplier: int):
    name = args.lr_scheduler
    num_warmup = args.lr_warmup_steps * step_multiplier
    num_training = args.max_train_steps * step_multiplier
    constant_steps = args.constant_steps * step_multiplier
    if name != "constant_with_warmup_cosine_decay":
        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_training,
        )
    return get_cosine_schedule_with_warmup_and_decay(
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training,
        constant_steps=constant_steps,
    )
