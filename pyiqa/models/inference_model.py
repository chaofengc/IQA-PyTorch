import torch
import torchvision as tv
from collections import OrderedDict
from os import path as osp
from PIL import Image

from pyiqa.archs import build_network
from pyiqa.losses import build_loss
from pyiqa.metrics import calculate_metric
from pyiqa.utils import get_root_logger, imwrite, tensor2img
from pyiqa.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


class InferenceModel():
    """Common model for quality inference of single image with default setting of each metric.""" 

    def __init__(self, 
                 metric_name,
                 metric_mode,
                 model_path=None,
                 img_range=1.0,
                 input_size=None,
                 mean=None,
                 std=None,
                 pre_process_x=None,
                 pre_process_y=None,
            ):
        super(InferenceModel, self).__init__()

        self.metric_mode = metric_mode

        # define network
        opt = {
                'type': metric_name,
                'pretrained_model_path': model_path,
                }
        self.net = build_network(opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)

        # load pretrained models
        #  if model_path is not None:
            #  self.net.load_pretrained_network(model_path)

        if pre_process_x is not None:
            self.trans_x = pre_process_x 
        if pre_process_y is not None:
            self.trans_y = pre_process_y 

        if pre_process_x is None and pre_process_y is None:
            tf_list = []
            tf_list.append(tv.transforms.ToTensor())
            if mean is not None and std is not None:
                tf_list.append(tv.transforms.Normalize(mean, std))
            self.trans_x = self.trans_y = tv.transforms.Compose(tf_list)
    
    def test(self, x, y=None):
        x = self.trans_x(x)
        x = x.unsqueeze(0).to(self.device)
        if self.metric_mode == 'FR':
            assert y is not None, 'Please specify reference image for Full Reference metric'
            y = self.trans_y(y)
            y = y.unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            if self.metric_mode == 'FR':
                output = self.net(x, y)
            elif self.metric_mode == 'NR':
                output = self.net(x)
        return output.cpu().item()


    
