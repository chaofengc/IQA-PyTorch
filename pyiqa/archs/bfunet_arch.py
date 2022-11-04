import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network

@ARCH_REGISTRY.register()
class BFUNet(nn.Module):
	"""UNet as defined in https://arxiv.org/abs/1805.07709"""
	def __init__(self, bias, residual_connection = False):
		super(BFUNet, self).__init__()
		self.conv1 = nn.Conv2d(1,32,5,padding = 2, bias = bias)
		self.conv2 = nn.Conv2d(32,32,3,padding = 1, bias = bias)
		self.conv3 = nn.Conv2d(32,64,3,stride=2, padding = 1, bias = bias)
		self.conv4 = nn.Conv2d(64,64,3,padding = 1, bias=bias)
		self.conv5 = nn.Conv2d(64,64,3,dilation=2, padding = 2, bias = bias)
		self.conv6 = nn.Conv2d(64,64,3,dilation = 4,padding = 4, bias = bias)
		self.conv7 = nn.ConvTranspose2d(64,64, 4,stride = 2, padding = 1, bias = bias)
		self.conv8 = nn.Conv2d(96,32,3,padding=1, bias = bias)
		self.conv9 = nn.Conv2d(32,1,5,padding = 2, bias = False)

		self.residual_connection = residual_connection;
		
	# @staticmethod
	# def add_args(parser):
	# 	"""Add model-specific arguments to the parser."""
	# 	parser.add_argument("--bias", action='store_true', help="use residual bias")
	# 	parser.add_argument("--residual", action='store_true', help="use residual connection")

	# @classmethod
	# def build_model(cls, args):
	# 	return cls(args.bias, args.residual)

	def forward(self, x):
		pad_right = x.shape[-2]%2
		pad_bottom = x.shape[-1]%2
		padding = nn.ZeroPad2d((0, pad_bottom,  0, pad_right))
		x = padding(x)

		out = F.relu(self.conv1(x))

		out_saved = F.relu(self.conv2(out))

		out = F.relu(self.conv3(out_saved))
		out = F.relu(self.conv4(out))
		out = F.relu(self.conv5(out))
		out = F.relu(self.conv6(out))
		out = F.relu(self.conv7(out))

		out = torch.cat([out,out_saved],dim = 1)

		out = F.relu(self.conv8(out))
		out = self.conv9(out)

		if self.residual_connection:
			out = x - out;

		if pad_bottom:
			out = out[:, :, :, :-1]
		if pad_right:
			out = out[:, :, :-1, :]

		return out
