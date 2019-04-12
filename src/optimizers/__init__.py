from helpers.config import ConfigClass
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

optim2class = {
    'sgd': SGD,
    'adam': Adam,
    'asgd': ASGD,
    'adamax': Adamax,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'rmsprop': RMSprop
}


def get_optimizer(optim_cfg: ConfigClass):
    optim_name = optim_cfg.name

    if optim_name is None:
        return Adam
    else:
        if optim_name not in optim2class:
            raise NotImplementedError(f"Optimizer {optim_name} not implemented")

        return optim2class[optim_name]
