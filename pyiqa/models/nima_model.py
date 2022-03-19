from collections import OrderedDict
import torch

from pyiqa.metrics import calculate_metric
from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel


@MODEL_REGISTRY.register()
class NIMAModel(GeneralIQAModel):
    """General module to train an IQA network."""

    def feed_data(self, data):
        self.img_input = data['img'].to(self.device)
        self.gt_mos = data['mos_label'].to(self.device)
        self.gt_mos_dist = data['mos_dist'].to(self.device)
        self.use_ref = False

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim']
        optim_params = [
            {
                'params': self.get_bare_model(self.net).base_model.parameters(),
                'lr': optim_opt.pop('lr_basemodel'),
            },
            {
                'params': self.get_bare_model(self.net).classifier.parameters(),
                'lr': optim_opt.pop('lr_classifier'),
            },
        ]

        optim_type = optim_opt.pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **optim_opt)
        self.optimizers.append(self.optimizer)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output_score = self.net(self.img_input, return_mos=True, return_dist=False)
        self.net.train()

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        self.output_mos, self.output_dist = self.net(self.img_input, return_mos=True, return_dist=True)

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_mos:
            l_mos = self.cri_mos(self.output_dist, self.gt_mos_dist)
            l_total += l_mos
            loss_dict['l_mos'] = l_mos

        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # log metrics in training batch
        pred_score = self.output_mos.squeeze(1).cpu().detach().numpy()
        gt_mos = self.gt_mos.squeeze(1).cpu().detach().numpy()
        for name, opt_ in self.opt['val']['metrics'].items():
            self.log_dict[f'train_metrics/{name}'] = calculate_metric([pred_score, gt_mos], opt_)
