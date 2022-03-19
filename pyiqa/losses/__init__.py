from copy import deepcopy

from pyiqa.utils import get_root_logger
from pyiqa.utils.registry import LOSS_REGISTRY
from .losses import CharbonnierLoss, L1Loss, MSELoss, WeightedTVLoss

from .iqa_losses import EMDLoss, PLCCLoss, NiNLoss

__all__ = ['L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'EMDLoss', 'PLCCLoss', 'NiNLoss']


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
