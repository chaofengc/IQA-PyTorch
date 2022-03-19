import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from pyiqa.archs import build_network
from pyiqa.losses import build_loss
from pyiqa.metrics import calculate_metric
from pyiqa.utils import get_root_logger, imwrite, logger, tensor2img
from pyiqa.utils.registry import MODEL_REGISTRY
from pyiqa.models import lr_scheduler as lr_scheduler
from .general_iqa_model import GeneralIQAModel


@MODEL_REGISTRY.register()
class DBCNNModel(GeneralIQAModel):
    """General module to train an IQA network."""

    def __init__(self, opt):
        super(DBCNNModel, self).__init__(opt)
        self.train_stage = 'train'

    def reset_optimizers_finetune(self):
        logger = get_root_logger()
        logger.info(f'\n Start finetune stage. Set all parameters trainable\n')
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net.named_parameters():
            v.requires_grad = True
            optim_params.append(v)

        optim_type = train_opt['optim_finetune'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim_finetune'])
        self.optimizers = [self.optimizer]

        # reset schedulers
        self.schedulers = []
        self.setup_schedulers('scheduler_finetune')

    def optimize_parameters(self, current_iter):
        if current_iter >= self.opt['train']['finetune_start_iter'] and self.train_stage != 'finetune':
            # copy best model from coarse training stage and reset optimizers
            self.copy_model(self.net_best, self.net)
            self.reset_optimizers_finetune()
            self.train_stage = 'finetune'

        super().optimize_parameters(current_iter)
