import argparse
import models
from util import util
from torch.autograd import Variable
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='vgg',use_gpu=opt.use_gpu)

im0 = util.load_image(opt.path0).transpose(2, 0, 1) / 255.
im1 = util.load_image(opt.path1).transpose(2, 0, 1) / 255.

# Load images
img0 = Variable(torch.FloatTensor(im0)[None,:,:,:])#util.im2tensor(im0) # RGB image from [-1,1]
img1 = Variable(torch.FloatTensor(im1)[None,:,:,:])#util.im2tensor(im1)
print(img0.size())

# Compute distance
dist01 = model.forward(img0,img1)
print(type(dist01))
print(dist01.item())
print('Distance: %.3f'%dist01)
