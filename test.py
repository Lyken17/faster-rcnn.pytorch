import _init_paths

import torch
from model.faster_rcnn.nasnet import nasnet

net = nasnet([1,2,3])
net._init_modules()
net.eval()

x = torch.zeros(1, 3, 800, 800)
print(net.RCNN_base)

# [torch.Size([24, 3, 600, 902]), torch.Size([24, 3]), torch.Size([24, 50, 5]), torch.Size([24])]
im_data = torch.randn(24, 3, 600, 902)
im_info = torch.randn(24, 3)
num_boxes = torch.randn(24, 50, 5)
gt_boxes = torch.randn(24)

dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
out = net(im_data, im_info, num_boxes, gt_boxes)
