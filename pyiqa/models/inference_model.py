import torch
import torchvision as tv

from pyiqa.archs import create_metric
from pyiqa.default_model_configs import DEFAULT_CONFIGS


class InferenceModel():
    """Common model for quality inference of single image with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            metric_mode,
            model_path=None,
            img_range=1.0,
            input_size=None,
            mean=None,
            std=None,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name
        metric_default_cfg = DEFAULT_CONFIGS[metric_name]
        if metric_name in DEFAULT_CONFIGS.keys():
            self.metric_mode = metric_default_cfg['metric_mode']
        else:
            self.metric_mode = metric_mode

        # load pretrained models
        if model_path is not None:
            kwargs['pretrained_model_path'] = model_path

        # define network
        self.net = create_metric(metric_name, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        self.net.eval()

        tf_list = []
        if input_size is not None:
            tf_list.append(tv.transforms.Resize(input_size))
        tf_list.append(tv.transforms.ToTensor())
        tf_list.append(tv.transforms.Lambda(lambda x: x * img_range))
        if mean is not None and std is not None:
            tf_list.append(tv.transforms.Normalize(mean, std))
        self.trans = tv.transforms.Compose(tf_list)

    def test(self, x, y=None):
        if not torch.is_tensor(x):
            x = self.trans(x)
            x = x.unsqueeze(0).to(self.device)
            if self.metric_mode == 'FR':
                assert y is not None, 'Please specify reference image for Full Reference metric'
                y = self.trans(y)
                y = y.unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.metric_mode == 'FR':
                output = self.net(x, y)
            elif self.metric_mode == 'NR':
                output = self.net(x)
        return output.cpu().item()
