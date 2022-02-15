import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
from PIL import Image
import torchvision as tv
import torch
from torch.autograd import Variable
import glob

import torch
import torchvision as tv

from pyiqa.archs import create_metric
from pyiqa.archs.arch_util import load_pretrained_network
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
            preprocess_x=None,
            preprocess_y=None,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name
        if metric_name in DEFAULT_CONFIGS.keys():
            self.metric_mode = DEFAULT_CONFIGS[metric_name]['metric_mode']
        else:
            self.metric_mode = metric_mode

        # define network
        self.net = create_metric(metric_name, **kwargs)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        self.net.eval()

        # load pretrained models
        if model_path is not None:
            load_pretrained_network(self.net, model_path)

        tf_list = []
        if input_size is not None:
            tf_list.append(tv.transforms.Resize(input_size))
        tf_list.append(tv.transforms.ToTensor())
        tf_list.append(tv.transforms.Lambda(lambda x: x * img_range))
        if mean is not None and std is not None:
            tf_list.append(tv.transforms.Normalize(mean, std))
        self.trans_x = self.trans_y = tv.transforms.Compose(tf_list)

        # This is only used to specific methods which has specific preprocessing, for example, ckdn
        if preprocess_x is not None and preprocess_y is not None:
            self.trans_x = preprocess_x
            self.trans_y = preprocess_y

    def test(self, x, y=None):
        if not torch.is_tensor(x):
            x = self.trans_x(x)
            x = x.unsqueeze(0).to(self.device)
            if self.metric_mode == 'FR':
                assert y is not None, 'Please specify reference image for Full Reference metric'
                y = self.trans_y(y)
                y = y.unsqueeze(0).to(self.device)
        if self.metric_mode == 'FR':
            output = self.net(x, y)
        elif self.metric_mode == 'NR':
            output = self.net(x)
        return output


def test_model(input, ref, model_name, metric_mode):
    # set up IQA model
    iqa_model = InferenceModel(model_name, metric_mode, None, 1.0, None, None,
                               None)
    metric_mode = iqa_model.metric_mode

    if os.path.isfile(input):
        input_paths = [input]
        if ref is not None:
            ref_paths = [ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(input, '*')))
        if ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(ref, '*')))

    for idx, img_path in enumerate(input_paths):
        img_name = os.path.basename(img_path)
        tar_img = Image.open(img_path)
        if metric_mode == 'FR':
            ref_img_path = ref_paths[idx]
            ref_img = Image.open(ref_img_path)
        else:
            ref_img = None
        score = iqa_model.test(tar_img, ref_img)
        print(f'{model_name} score of {img_name} is: {score}')


def prepare_image(image):
    preprocess_y = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 1),
    ])
    image = preprocess_y(image)
    squeezeImage = image.unsqueeze(0)
    return squeezeImage


def test_backward(dist1_path, dist2_path, model_name, metric_mode):
    dis1 = prepare_image(Image.open(dist1_path).convert('RGB'))
    dis2 = prepare_image(Image.open(dist2_path).convert('RGB'))

    iqa_model = InferenceModel(model_name, metric_mode, None, 1.0, None, None,
                               None)
    metric_mode = iqa_model.metric_mode

    if metric_mode == "FR":
        input1 = Variable(dis1, requires_grad=True)
        input2 = Variable(dis2, requires_grad=True)
        score = iqa_model.test(input1, input2)
        score.requires_grad_()
        score.mean().backward()
        res = input1.grad
        if input1.grad is not None:
            print(
                f'{model_name} suppurts backward with result shape {res.shape}'
            )
        print(score)

    else:
        input = torch.zeros((2, 3, 384, 512))
        input[0, ...] = dis1
        input[1, ...] = dis2
        batch = Variable(input, requires_grad=True)
        score = iqa_model.test(batch)
        score.requires_grad_()
        score.mean().backward()
        res = batch.grad
        if res is not None:
            print(
                f'{model_name} suppurts backward with result shape {res.shape}'
            )
        print(score)


if __name__ == '__main__':
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "lpips", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "dists", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "ckdn", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "fsim", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "ssim", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "ms_ssim", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "cw_ssim", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "psnr", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "vif", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "gmsd", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "nlpd", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "vsi", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "mad", "FR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "musiq", "NR")
    # # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "dbcnn", "NR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "brisque", "NR")
    # test_model("./ResultsCalibra/dist_dir", "./ResultsCalibra/ref_dir", "niqe", "NR")

    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "lpips", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "dists", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "ckdn", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "fsim", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "ssim", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "ms_ssim", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "cw_ssim", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "psnr", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "vif", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "gmsd", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "nlpd", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "vsi", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/ref_dir/I03.bmp", "mad", "FR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/dist_dir/I19.bmp", "musiq", "NR")
    test_backward("./ResultsCalibra/dist_dir/I03.bmp", "./ResultsCalibra/dist_dir/I19.bmp", "dbcnn", "NR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/dist_dir/I19.bmp", "brisque", "NR")
    # test_backward("./ResultsCalibra/dist_dir/I03.bmp",
    #               "./ResultsCalibra/dist_dir/I19.bmp", "niqe", "NR")
