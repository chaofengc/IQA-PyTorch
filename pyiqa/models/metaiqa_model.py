import torch
from collections import OrderedDict
from pyiqa.models.base_model import BaseModel
from pyiqa.utils.registry import MODEL_REGISTRY
from pyiqa.archs import build_network


@MODEL_REGISTRY.register()
class MetaIQAModel(BaseModel):
    def __init__(self, opt):
        super(MetaIQAModel, self).__init__(opt)
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = torch.nn.MSELoss().to(self.device)
        base_params = [p for n, p in self.net_g.named_parameters() if 'head' not in n]
        fc_params = [p for n, p in self.net_g.named_parameters() if 'head' in n]
        lr_backbone = train_opt.get('lr_resnet_layer', train_opt['lr'])
        lr_fc = train_opt.get('lr_head', 1e-2)

        self.optimizers.append(
            torch.optim.Adam([
                {'params': base_params, 'lr': float(lr_backbone)},
                {'params': fc_params, 'lr': float(lr_fc)}
            ], weight_decay=train_opt.get('weight_decay', 0))
        )

    def feed_data(self, data):
        self.img = data['target_img'].to(self.device)
        self.label = data['mos'].to(self.device).float().view(-1, 1)

    def optimize_parameters(self, current_iter):
        optimizer = self.optimizers[0]
        optimizer.zero_grad()
        self.output_score = self.net_g(self.img)
        loss = self.criterion(self.output_score, self.label)
        loss.backward()
        optimizer.step()
        self.log_dict = OrderedDict(loss=loss.item())