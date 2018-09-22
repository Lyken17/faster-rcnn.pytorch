from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from model.utils.config import cfg
# from model.faster_rcnn.faster_rcnn import _fasterRCNN
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import math
import torch.utils.model_zoo as model_zoo
# import pdb

model_urls = {'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
              'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
              'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth', }

import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
	return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
	return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expand_ratio)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(  # dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),  # pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), )
		else:
			self.conv = nn.Sequential(  # pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),  # pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), )

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


class MobileNetV2(nn.Module):
	def __init__(self, n_class=1000, input_size=224, width_mult=1.):
		super(MobileNetV2, self).__init__()
		block = InvertedResidual
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [  # t, c, n, s
			[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1], ]

		# building first layer
		assert input_size % 32 == 0
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.features = [conv_bn(3, input_channel, 2)]
		# building inverted residual blocks
		for t, c, n, s in interverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				if i == 0:
					self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
				else:
					self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
				input_channel = output_channel
		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))
		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		# building classifier
		self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, n_class), )

		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.mean(3).mean(2)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


def mobilenetv2(pretrained=False):
	model = MobileNetV2()
	if pretrained:
		url = "http://file.lzhu.me/pytorch-model-zoo/mobilenet_v2.pth.tar"
		model.load_state_dict(model_zoo.load_url(url))
	return model


if __name__ == '__main__':
	net = mobilenetv2()
	# print(net)
	# print(net.features)
	print(net.features[1])


# class mobilenet(_fasterRCNN):
# 	def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
# 		self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
# 		self.dout_base_model = 1024
# 		self.pretrained = pretrained
# 		self.class_agnostic = class_agnostic
#
# 		_fasterRCNN.__init__(self, classes, class_agnostic)
#
# 	def _init_modules(self):
# 		resnet = mobilenetv2(pretrained=self.pretrained)
#
# 		if self.pretrained == True:
# 			print("Loading pretrained weights from %s" % (
# 				self.model_path))  # state_dict = torch.load(self.model_path)  #  resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})
#
# 		# Build resnet.
# 		self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
# 			resnet.layer2, resnet.layer3)
#
# 		self.RCNN_top = nn.Sequential(resnet.layer4)
#
# 		self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
# 		if self.class_agnostic:
# 			self.RCNN_bbox_pred = nn.Linear(2048, 4)
# 		else:
# 			self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)
#
# 		# Fix blocks
# 		for p in self.RCNN_base[0].parameters(): p.requires_grad = False
# 		for p in self.RCNN_base[1].parameters(): p.requires_grad = False
#
# 		assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
# 		if cfg.RESNET.FIXED_BLOCKS >= 3:
# 			for p in self.RCNN_base[6].parameters(): p.requires_grad = False
# 		if cfg.RESNET.FIXED_BLOCKS >= 2:
# 			for p in self.RCNN_base[5].parameters(): p.requires_grad = False
# 		if cfg.RESNET.FIXED_BLOCKS >= 1:
# 			for p in self.RCNN_base[4].parameters(): p.requires_grad = False
#
# 		def set_bn_fix(m):
# 			classname = m.__class__.__name__
# 			if classname.find('BatchNorm') != -1:
# 				for p in m.parameters(): p.requires_grad = False
#
# 		self.RCNN_base.apply(set_bn_fix)
# 		self.RCNN_top.apply(set_bn_fix)
#
# 	def train(self, mode=True):
# 		# Override train so that the training mode is set as we want
# 		nn.Module.train(self, mode)
# 		if mode:
# 			# Set fixed blocks to be in eval mode
# 			self.RCNN_base.eval()
# 			self.RCNN_base[5].train()
# 			self.RCNN_base[6].train()
#
# 			def set_bn_eval(m):
# 				classname = m.__class__.__name__
# 				if classname.find('BatchNorm') != -1:
# 					m.eval()
#
# 			self.RCNN_base.apply(set_bn_eval)
# 			self.RCNN_top.apply(set_bn_eval)
#
# 	def _head_to_tail(self, pool5):
# 		fc7 = self.RCNN_top(pool5).mean(3).mean(2)
# 		return fc7
