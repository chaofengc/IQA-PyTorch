from collections import OrderedDict

from pyiqa.metrics import calculate_metric
from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel


@MODEL_REGISTRY.register()
class WaDIQaMModel(GeneralIQAModel):
    """General module to train an IQA network."""

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim']
        bare_net = self.get_bare_model(self.net)
        optim_params = [
            {
                'params': bare_net.features.parameters(),
                'lr': optim_opt.pop('lr_basemodel'),
            },
            {
                'params': [p for k, p in bare_net.named_parameters() if 'features' not in k],
                'lr': optim_opt.pop('lr_fc_layers'),
            },
        ]

        optim_type = optim_opt.pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **optim_opt)
        self.optimizers.append(self.optimizer)
