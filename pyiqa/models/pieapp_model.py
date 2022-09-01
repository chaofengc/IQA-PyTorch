from collections import OrderedDict
import torch
from pyiqa.metrics.correlation_coefficient import calculate_rmse
from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


@MODEL_REGISTRY.register()
class PieAPPModel(GeneralIQAModel):
    """General module to train an IQA network."""

    def feed_data(self, data):
        is_test = 'img' in data.keys() 

        if 'use_ref' in self.opt['train']:
            self.use_ref = self.opt['train']['use_ref']

        if is_test:
            self.img_input = data['img'].to(self.device)
            self.gt_mos = data['mos_label'].to(self.device)
            self.ref_input = data['ref_img'].to(self.device)
            self.ref_img_path = data['ref_img_path']
            self.img_path = data['img_path']
        else:
            self.img_A_input = data['distA_img'].to(self.device)
            self.img_B_input = data['distB_img'].to(self.device)
            self.img_ref_input = data['ref_img'].to(self.device)
            self.gt_prob = data['mos_label'].to(self.device)

            # from torchvision.utils import save_image
            # save_image(torch.cat([self.img_A_input, self.img_B_input, self.img_ref_input], dim=0), 'tmp_test_pieappdataset.jpg')
            # exit()

    def optimize_parameters(self, current_iter):
        
        self.optimizer.zero_grad()
        
        score_A = self.net(self.img_A_input, self.img_ref_input)
        score_B = self.net(self.img_B_input, self.img_ref_input)
        train_output_score = (1 / (1 + torch.exp(score_A - score_B)))
        
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_mos:
            l_mos = self.cri_mos(train_output_score, self.gt_prob)
            l_total += l_mos
            loss_dict['l_mos'] = l_mos

        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # log metrics in training batch
        pred_score = train_output_score.squeeze(-1).cpu().detach().numpy()
        gt_prob = self.gt_prob.squeeze(-1).cpu().detach().numpy()
        
        self.log_dict[f'train_metrics/rmse'] = calculate_rmse(pred_score, gt_prob)