from collections import OrderedDict
import torch
from pyiqa.metrics import calculate_metric 
from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp


@MODEL_REGISTRY.register()
class BAPPSModel(GeneralIQAModel):
    """General module to train an IQA network."""

    def feed_data(self, data):

        if 'use_ref' in self.opt['train']:
            self.use_ref = self.opt['train']['use_ref']

        self.img_A_input = data['distA_img'].to(self.device)
        self.img_B_input = data['distB_img'].to(self.device)
        self.img_ref_input = data['ref_img'].to(self.device)
        self.gt_mos = data['mos_label'].to(self.device)
        self.img_path = data['img_path']

        # from torchvision.utils import save_image
        # print(self.img_ref_input.shape)
        # save_image(torch.cat([self.img_ref_input, self.img_A_input, self.img_B_input], dim=0), 'tmp_test_bappsdataset.jpg')
        # exit()
    
    def compute_accuracy(self, d0, d1, judge):
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        acc = d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)
        return acc.mean()

    def optimize_parameters(self, current_iter):
        
        self.optimizer.zero_grad()
        
        score_A = self.net(self.img_A_input, self.img_ref_input)
        score_B = self.net(self.img_B_input, self.img_ref_input)
        # For BAPPS, 
        train_output_score = (1 / (1 + torch.exp(score_B - score_A)))
        
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_mos:
            l_mos = self.cri_mos(train_output_score, self.gt_mos)
            l_total += l_mos
            loss_dict['l_mos'] = l_mos

        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # log metrics in training batch
        
        self.log_dict[f'train_metrics/acc'] = self.compute_accuracy(score_A, score_B, self.gt_mos)

    @torch.no_grad()
    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.score_A = self.net(self.img_A_input, self.img_ref_input)
            self.score_B = self.net(self.img_B_input, self.img_ref_input)
        self.net.train()

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        pred_score_A = []
        pred_score_B = []
        gt_mos = []
        for idx, val_data in enumerate(dataloader):
            img_name = osp.basename(val_data['img_path'][0])
            self.feed_data(val_data)
            self.test()
            if len(self.score_A.shape) <= 1:
                self.score_A = self.score_A.reshape(-1, 1)
                self.score_B = self.score_B.reshape(-1, 1)
            pred_score_A.append(self.score_A)
            pred_score_B.append(self.score_B)
            gt_mos.append(self.gt_mos)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name:>20}')
        if use_pbar:
            pbar.close()

        pred_score_A = torch.cat(pred_score_A, dim=0).squeeze(1).cpu().numpy()
        pred_score_B = torch.cat(pred_score_B, dim=0).squeeze(1).cpu().numpy()
        gt_mos = torch.cat(gt_mos, dim=0).squeeze(1).cpu().numpy()

        if with_metrics:
            # calculate all metrics 
            for name, opt_ in self.opt['val']['metrics'].items():
                self.metric_results[name] = calculate_metric([pred_score_A, pred_score_B, gt_mos], opt_)
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net, self.net_best)
                    self.save_network(self.net_best, 'net_best')
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated 
                if sum(updated): 
                    self.copy_model(self.net, self.net_best)
                    self.save_network(self.net_best, 'net_best')
            
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
