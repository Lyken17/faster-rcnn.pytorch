import math

from .modules.layers import *
from .utils import load_url, download_url

model_urls = {'mobilenetv2': {"weight": "http://file.lzhu.me/hancai/74.3/model.pth.tar",
	"config": "http://file.lzhu.me/hancai/74.3/net.config"}}


class MobileInvertedResidualBlock(BasicUnit):

	def __init__(self, mobile_inverted_conv, shortcut):
		super(MobileInvertedResidualBlock, self).__init__()

		self.mobile_inverted_conv = mobile_inverted_conv
		self.shortcut = shortcut

	def forward(self, x):
		if self.mobile_inverted_conv.is_zero_layer():
			return x
		elif self.shortcut is None or self.shortcut.is_zero_layer():
			return self.mobile_inverted_conv(x)

		conv_x = self.mobile_inverted_conv(x)
		skip_x = self.shortcut(x)

		conv_channels = conv_x.size()[1]
		skip_channels = skip_x.size()[1]
		padding_channels = abs(conv_channels - skip_channels)
		if padding_channels > 0:
			batch_size, _, h, w = conv_x.size()
			if x.is_cuda:
				with torch.cuda.device(x.get_device()):
					padding = torch.cuda.FloatTensor(batch_size, padding_channels, h, w).fill_(0)
			else:
				padding = torch.zeros(batch_size, padding_channels, h, w)
			padding = torch.autograd.Variable(padding, requires_grad=False)
			if conv_channels > skip_channels:
				skip_x = torch.cat((skip_x, padding), 1)
			else:
				conv_x = torch.cat((conv_x, padding), 1)
		return skip_x + conv_x

	@property
	def unit_str(self):
		return '(%s, %s)' % (
			self.mobile_inverted_conv.unit_str, self.shortcut.unit_str if self.shortcut is not None else None)

	@property
	def config(self):
		return {'name': MobileInvertedResidualBlock.__name__, 'mobile_inverted_conv': self.mobile_inverted_conv.config,
		        'shortcut': self.shortcut.config if self.shortcut is not None else None, }

	@staticmethod
	def build_from_config(config):
		mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
		shortcut = set_layer_from_config(config['shortcut'])
		return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

	def get_flops(self, x):
		flops1, _ = self.mobile_inverted_conv.get_flops(x)
		if self.shortcut:
			flops2, _ = self.shortcut.get_flops(x)
		else:
			flops2 = 0

		return flops1 + flops2, self.forward(x)


class MobileNets(BasicUnit):

	def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
		super(MobileNets, self).__init__()

		self.first_conv = first_conv
		self.blocks = nn.ModuleList(blocks)
		self.feature_mix_layer = feature_mix_layer
		self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = classifier

	def forward(self, x):
		x = self.first_conv(x)
		for block in self.blocks:
			x = block(x)
		if self.feature_mix_layer:
			x = self.feature_mix_layer(x)
		x = self.global_avg_pooling(x)
		x = x.view(x.size(0), -1)  # flatten
		x = self.classifier(x)
		return x

	@property
	def unit_str(self):
		raise ValueError('not needed')

	@property
	def config(self):
		return {'name': MobileNets.__name__, 'first_conv': self.first_conv.config,
		        'feature_mix_layer': self.feature_mix_layer.config if self.feature_mix_layer is not None else None,
		        'classifier': self.classifier.config, 'blocks': [block.config for block in self.blocks], }

	@staticmethod
	def build_from_config(config):
		first_conv = set_layer_from_config(config['first_conv'])
		feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
		classifier = set_layer_from_config(config['classifier'])
		blocks = []
		for block_config in config['blocks']:
			blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

		return MobileNets(first_conv, blocks, feature_mix_layer, classifier)

	def get_flops(self, x):
		flop, x = self.first_conv.get_flops(x)

		for block in self.blocks:
			delta_flop, x = block.get_flops(x)
			flop += delta_flop
		if self.feature_mix_layer:
			delta_flop, x = self.feature_mix_layer.get_flops(x)
			flop += delta_flop
		x = self.global_avg_pooling(x)
		x = x.view(x.size(0), -1)  # flatten

		delta_flop, x = self.classifier.get_flops(x)
		flop += delta_flop
		return flop, x

	def set_bn_param(self, bn_momentum, bn_eps):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.momentum = bn_momentum
				m.eps = bn_eps
		return

	def init_model(self, model_init, init_div_groups=True):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if model_init == 'he_fout':
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					if init_div_groups:
						n /= m.groups
					m.weight.data.normal_(0, math.sqrt(2. / n))
				elif model_init == 'he_fin':
					n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
					if init_div_groups:
						n /= m.groups
					m.weight.data.normal_(0, math.sqrt(2. / n))
				else:
					raise NotImplementedError
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def weight_parameters(self):
		return self.parameters()

	@staticmethod
	def _make_divisible(v, divisor, min_val=None):
		"""
		This function is taken from the original tf repo.
		It ensures that all layers have a channel number that is divisible by 8
		It can be seen here:
		https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
		:param v:
		:param divisor:
		:param min_val:
		:return:
		"""
		if min_val is None:
			min_val = divisor
		new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
		# Make sure that round down does not go down by more than 10%.
		if new_v < 0.9 * v:
			new_v += divisor
		return new_v

	def adjust_width(self, width_type, width_mult):
		if len(self.blocks) == 17:
			if width_type == 'mnas':
				cell_stages = [1, 3, 3, 3, 2, 4, 1]
			elif width_type == 'v2':
				cell_stages = [1, 2, 3, 4, 3, 3, 1]
			else:
				raise NotImplementedError
		elif len(self.blocks) == 22:
			cell_stages = [1, 4, 4, 4, 4, 4, 1]
		else:
			raise NotImplementedError

		if width_type == 'mnas':
			width_stages = [16, 24, 40, 80, 96, 192, 320]
		else:
			width_stages = [16, 24, 32, 64, 96, 160, 320]

		last_channel = self.classifier.in_features

		self.first_conv.out_channels = self._make_divisible(32 * width_mult, 8)
		input_channel = self.first_conv.out_channels

		pointer = 0
		for n, c in zip(cell_stages, width_stages):
			for i in range(pointer, pointer + n):
				output_channel = self._make_divisible(c * width_mult, 8)
				if not isinstance(self.blocks[i].mobile_inverted_conv, ZeroLayer):
					self.blocks[i].mobile_inverted_conv.in_channels = input_channel
					self.blocks[i].mobile_inverted_conv.out_channels = output_channel
				else:
					output_channel = input_channel

				if self.blocks[i].shortcut is not None and not isinstance(self.blocks[i].shortcut, ZeroLayer):
					self.blocks[i].shortcut.in_channels = input_channel
					self.blocks[i].shortcut.out_channels = input_channel
				else:
					pass
				input_channel = output_channel
			pointer += n
		if self.feature_mix_layer is not None:
			self.feature_mix_layer.in_channels = input_channel
			self.feature_mix_layer.out_channels = last_channel
		else:
			last_channel = input_channel

		self.classifier.in_features = last_channel

		return self.build_from_config(self.config)


class MobileNetV2(MobileNets):

	def __init__(self, n_classes=1000, width_mult=1, bn_param=(0.1, 1e-5), dropout_rate=0.2, first_block_relu=False,
	             no_mix_layer=False, skip_when_channel_not_match=False, shortcut_downsample=None):

		input_channel = 32
		if no_mix_layer:
			last_channel = 320
			interverted_residual_setting = [  # t, c, n, s
				[1, 16, 1, 1], [6, 24, 2, 2], [6, 40, 3, 2], [6, 80, 4, 2], [6, 96, 3, 1], [6, 192, 3, 2],
				[6, 320, 1, 1], ]
		else:
			last_channel = 1280
			interverted_residual_setting = [  # t, c, n, s
				[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
				[6, 320, 1, 1], ]

		input_channel = self._make_divisible(input_channel * width_mult, 8)
		last_channel = self._make_divisible(last_channel * width_mult, 8) if width_mult > 1.0 else last_channel

		# first conv layer
		first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6',
		                       ops_order='weight_bn_act')
		# inverted residual blocks
		blocks = []
		for t, c, n, s in interverted_residual_setting:
			output_channel = self._make_divisible(c * width_mult, 8)
			if t == 1:
				mb_final_relu = first_block_relu
			else:
				mb_final_relu = False
			for i in range(n):
				if i == 0:
					stride = s
				else:
					stride = 1
				mobile_inverted_conv = MBInvertedConvLayer(in_channels=input_channel, out_channels=output_channel,
				                                           kernel_size=3, stride=stride, expand_ratio=t,
				                                           has_final_relu=mb_final_relu, )
				if stride == 1:
					if input_channel == output_channel:
						shortcut = IdentityLayer(input_channel, input_channel)
					elif input_channel < output_channel and skip_when_channel_not_match:
						shortcut = IdentityLayer(input_channel, input_channel)
					else:
						shortcut = None
				elif shortcut_downsample == 'max_pool':
					shortcut = PoolingLayer(input_channel, input_channel, 'max', kernel_size=stride, stride=stride)
				elif shortcut_downsample == 'avg_pool':
					shortcut = PoolingLayer(input_channel, input_channel, 'avg', kernel_size=stride, stride=stride)
				else:
					shortcut = None
				blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
				input_channel = output_channel
		if no_mix_layer:
			feature_mix_layer = None
		else:
			# 1x1_conv before global average pooling
			feature_mix_layer = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6',
			                              ops_order='weight_bn_act', )

		classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

		super(MobileNetV2, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

		# set bn param
		self.set_bn_param(bn_momentum=bn_param[0], bn_eps=bn_param[1])


def mobilenet_v2(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on Places
	"""
	import json
	res = download_url(model_urls["mobilenetv2"]["config"])
	with open(res, "r") as fp:
		cfg = json.load(fp)
		net = MobileNetV2.build_from_config(cfg)
		if pretrained:
			net.load_state_dict(load_url(model_urls["mobilenetv2"]["weight"]), strict=False)
	return net


if __name__ == '__main__':
	net = mobilenet_v2(pretrained=True)
	print(net)
	exit(0)

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN


class nasnet(_fasterRCNN):
	def __init__(self, classes, pretrained=False, class_agnostic=False):
		# self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
		self.pretrained = pretrained
		self.class_agnostic = class_agnostic
		self.dout_base_model = 432 # may change under different settings
		self.dout_top_model = 432 * 4

		_fasterRCNN.__init__(self, classes, class_agnostic)

		self.mobilenet = mobilenet_v2(pretrained=self.pretrained)
		# self.dout_base_model = self.mobilenet.feature_mix_layer.in_channels
		# self.dout_top_model = self.mobilenet.feature_mix_layer.in_channels * 4

	def _init_modules(self):

		if self.pretrained == True:
			print("Loading pretrained weights from file.lzhu.me")

		# Build resnet.
		# self.RCNN_base = nn.Sequential(*list(self.mobilenet.children())[:12])
		# self.RCNN_top  = nn.Sequential(*list(self.mobilenet.children())[12:])

		# self.RCNN_base = nn.Sequential(*list(self.mobilenet.features._modules.values()))
		# self.RCNN_top  = self.mobilenet.classifier

		self.RCNN_base = nn.Sequential(
			*([self.mobilenet.first_conv, ] + list(self.mobilenet.blocks.children()))
		)
		# self.RCNN_top = nn.Sequential(*list(self.mobilenet.feature_mix_layer.children()))
		self.RCNN_top = self.mobilenet.feature_mix_layer

		# self.RCNN_cls_score = nn.Linear(self.dout_top_model, self.n_classes)
		# if self.class_agnostic:
		# 	self.RCNN_bbox_pred = nn.Linear(self.dout_top_model, 4)
		# else:
		# 	self.RCNN_bbox_pred = nn.Linear(self.dout_top_model, 4 * self.n_classes)
		n = self.dout_top_model
		self.RCNN_cls_score = nn.Linear(n, self.n_classes)
		if self.class_agnostic:
			self.RCNN_bbox_pred = nn.Linear(n, 4)
		else:
			self.RCNN_bbox_pred = nn.Linear(n, 4 * self.n_classes)
		# Fix blocks
		assert (0 <= cfg.MOBILENET.FIXED_LAYERS <= 12)
		for m in list(self.RCNN_base.children())[:cfg.MOBILENET.FIXED_LAYERS]:
			for p in m.parameters():
				p.requires_grad = False

		def set_bn_fix(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				for p in m.parameters(): p.requires_grad = False

		self.RCNN_base.apply(set_bn_fix)
		self.RCNN_top.apply(set_bn_fix)

	def train(self, mode=True):
		# Override train so that the training mode is set as we want
		nn.Module.train(self, mode)
		if mode:
			# Set fixed blocks to be in eval mode
			# self.RCNN_base.eval()
			# self.RCNN_base[5].train()
			# self.RCNN_base[6].train()
			for m in list(self.RCNN_base.children())[:cfg.MOBILENET.FIXED_LAYERS]:
				for p in m.parameters():
					p.requires_grad = False

			def set_bn_eval(m):
				classname = m.__class__.__name__
				if classname.find('BatchNorm') != -1:
					m.eval()

			self.RCNN_base.apply(set_bn_eval)
			self.RCNN_top.apply(set_bn_eval)

	def _head_to_tail(self, pool5):
		fc7 = self.RCNN_top(pool5).mean(3).mean(2)
		# print("pool:\t", pool5.size())
		# print("fc7:\t", fc7.size())
		return fc7

