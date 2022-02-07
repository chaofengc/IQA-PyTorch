import torch
import torch.nn as nn
import timm
from pyiqa.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class NIMA(nn.Module):
    """Neural IMage Assessment model proposed by 

    Talebi, Hossein, and Peyman Milanfar. 
    "NIMA: Neural image assessment." 
    IEEE transactions on image processing 27, no. 8 (2018): 3998-4011.

    Modification: 
        - for simplicity, we use global average pool for all models
        - we remove the dropout, because parameters with avg pool is much less.

    Args:
        base_model_name: pretrained model to extract features, can be any models supported by timm.
                         Models used in the paper: vgg16, inception_resnet_v2, mobilenetv2_100 

        default input shape:
            - vgg and mobilenet: (N, 3, 224, 224)
            - inception: (N, 3, 299, 299)
    """
    def __init__(self, 
                 base_model_name='vgg16', 
                 num_classes=10, 
                 dropout_rate=0.,
                 default_mean=[0.485, 0.456, 0.406],
                 default_std=[0.229, 0.224, 0.225],
                 ):
        super(NIMA, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True, features_only=True)

        # if 'vgg' in base_model_name:
        #     self.global_pool = nn.Flatten()
        #     in_ch = 7 * 7 * 512
        #     p_dropout = 0.75
        # else:
        #     self.global_pool = nn.AdaptiveAvgPool2d(1)
        #     in_ch = self.base_model.feature_info.channels()[-1]
        #     p_dropout = 0.0

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        in_ch = self.base_model.feature_info.channels()[-1]
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=in_ch, out_features=num_classes),
            nn.Softmax(dim=1)
            )

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

    def _get_mos_from_dist(self, pred_score):
        pred_score = pred_score * torch.arange(1, self.num_classes + 1).to(pred_score).unsqueeze(0)
        pred_score = pred_score.sum(dim=1, keepdim=True)
        return pred_score

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def forward(self, x, return_mos=True, return_dist=False):
        # imagenet normalization of input is hard coded 
        x = self.preprocess(x)
        x = self.base_model(x)[-1]
        x = self.global_pool(x)
        dist = self.classifier(x)
        mos = self._get_mos_from_dist(dist)
        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(dist)

        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]

if __name__ == '__main__':
    import torch
    x = torch.randn(2, 3, 224, 224)
    model = NIMA('vgg16')
    model(x)
    model = NIMA('inception_resnet_v2')
    model(x)
    model = NIMA('mobilenetv2_100')
    model(x)