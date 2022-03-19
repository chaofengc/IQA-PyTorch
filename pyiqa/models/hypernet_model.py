from collections import OrderedDict
import torch

from pyiqa.metrics import calculate_metric
from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel


@MODEL_REGISTRY.register()
class HyperNetModel(GeneralIQAModel):
    """General module to train an IQA network."""

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output_score = self.get_bare_model(self.net).random_crop_test(self.img_input)
        self.net.train()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim']
        bare_net = self.get_bare_model(self.net)
        optim_params = [
            {
                'params': bare_net.base_model.parameters(),
                'lr': optim_opt.pop('lr_basemodel'),
            },
            {
                'params': [p for k, p in bare_net.named_parameters() if 'base_model' not in k],
                'lr': optim_opt.pop('lr_hypermodule'),
            },
        ]

        optim_type = optim_opt.pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **optim_opt)
        self.optimizers.append(self.optimizer)
