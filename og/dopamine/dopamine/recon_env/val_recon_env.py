import os, sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import autograd
import gym.spaces

sys.path.append('/home/davidyzeng/machinelearning/cones/python/packages/mrirecon')
sys.path.append('/home/davidyzeng/recon_pytorch')
import class_gan_unrolled

import bartwrap
import torchvision

###################
# For reconstruction environments, the API is a torch.Tensor.cuda()
# im: [256,256,2]
# sense: [1,8,256,256]
# mask: [1,256,256]
# returns: [256,256,2]

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-1,-2)), norm='ortho'), axes=(-1,-2))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x, axes=(-1,-2)), norm='ortho'), axes=(-1,-2))

class unrolled_recon():
 
    # expects im[1,2,256,256]
    # sense [1,8,256,256]
    # mask [1,256,256]
    def __init__(self,filepath=None):
        if filepath == None:
            filepath = '/mnt/dense/grace/20180726_2136/350000.ckpt'
        self.recon_net = class_gan_unrolled.Unrolled([256,256], unroll_layers=5, res_blocks=3, res_layers=2)
        self.device = 'cuda:0'
        recon_net_ckpt = torch.load(filepath)
        self.recon_net.load_state_dict(recon_net_ckpt['state_dict'])
        self.recon_net.cuda()
        self.recon_net.eval()

    def __call__(self, im, sense, mask):
        im = torch.Tensor(np.expand_dims(np.transpose(im,(2,0,1)),axis=0)).to(self.device)
        mask = torch.Tensor(mask).to(self.device)
        sense = np.transpose(np.expand_dims(sense, axis=0),(0,2,3,4,1))
        sense = torch.Tensor(sense).to(self.device)
        y = self.recon_net(im, sense, mask)
        y = y.cpu().detach().numpy() # [1,2,256,256]
        y = y[0,:,:,:] + 1j*y[0,:,:,:]
        y = np.transpose(y,(1,2,0))
        del im, mask, sense
        return y

class cs_recon():

    def __init__(self):
        pass

    def __call__(self, im, sense, mask):
        #ii = im.cpu().numpy()
        #ii = ii[0,0,:,:] + 1j*ii[0,1,:,:]
        ii = im[:,:,0] + 1j*im[:,:,1]
        kk = fft2c(ii)
        y_hat = bartwrap.bart_cs(kk)
        y_hat = np.stack((y_hat.real,y_hat.imag),axis=-1)
        #y_hat = torch.Tensor(y_hat).cuda()
        return y_hat

class fft_recon():

    def __init__(self):
        pass

    def __call__(self, im, sense, mask):
        return im

###################
# For rewards, the API is a [256,256,2] numpy array, single number output

class discriminator_reward():

    def __init__(self,filepath=None):
        if filepath == None:
            filepath = '/media/cineraid/davidyzeng/recon_pytorch_runs/gan/20180808_0129/5000.ckpt'
        self.reward_net = class_gan_unrolled.Discriminator()
        reward_net_ckpt = torch.load(filepath)
        self.reward_net.load_state_dict(reward_net_ckpt['state_dictD'])
        self.reward_net.cuda()
        self.reward_net.eval()

    def __call__(self, im):
        im = torch.Tensor(np.expand_dims(np.transpose(im,(2,0,1)),axis=0)).cuda()
        return self.reward_net(im).item()

class L2_reward():

    def __init__(self):
        pass

    def __call__(self, im):
        return np.sqrt(np.sum(np.linalg.norm(im,ord=2,axis=-1)**2))/im.size
        #return torch.sqrt(torch.sum(torch.norm(im[0,:,:,:],p=2,dim=0)**2))/im.numel()

class L1_reward():

    def __init__(self):
        pass

    def __call__(self, im):
        return np.sum(np.linalg.norm(im,ord=2,axis=-1))/im.size
        #return torch.sum(torch.norm(im[0,:,:,:],p=2,dim=0))/im.numel()

###################

class recon_env():

  def __init__(self, recon, reward, base_dir,R=2):

    self.curr_img = np.zeros((1,256,256,2))
    self.sampled_lines = np.zeros((256,),dtype=np.float32)
    self.mask = np.zeros((256,256),dtype=np.float32)

    data_file = '/home_local/grace/pytorch_data.h5'
    f = h5.File(data_file,'r')
    self.kspace = f['validate_kspace']
    self.sense = f['validate_sense']

    self.sensemap_data = None
    self.Y_Y = None
    self.mask_data = None

    self.recon = recon
    self.reward = reward

    #self.reward_history = np.zeros((256,)) #qqq
    self.line_order = np.zeros((256,))
    self.reward_order = np.zeros((256,))

    self.action_space = gym.spaces.Discrete(256)
    self.game_over = False
    self.sample_image = np.zeros((256,256))

    self.base_dir = base_dir
    logdir = os.path.join(base_dir,'imglog')
    self.total_done = 0

  def reset(self):

    del self.sensemap_data, self.Y_Y

    idx = np.random.randint(self.kspace.shape[0])
    idx = 83
    print(idx)
    #idx = 2000 #qqq
    self.curr_img = 0*self.curr_img
    self.sampled_lines = 0*self.sampled_lines
    self.mask = 0*self.mask

    complex_kspace = self.kspace[idx,0:8,:,:] + 1j*self.kspace[idx,8:,:,:]
    complex_kspace = complex_kspace/(np.amax(np.abs(complex_kspace))+1e-6)
    complex_im = ifft2c(complex_kspace)
    complex_sense = self.sense[idx,0:8,:,:] + 1j*self.sense[idx,8:,:,:]
    self.im0 = complex_im*np.conj(complex_sense)
    self.im0 = np.sum(self.im0.real,axis=0) # grace added
    y_im = np.stack((self.im0.real,self.im0.imag),axis=-1).astype(np.float32)
    #y_im = np.expand_dims(y_im,0)
    self.sensemap = np.stack((complex_sense.real,complex_sense.imag),axis=-1)
    self.sensemap = np.transpose(self.sensemap, (0,3,1,2))
    #self.sensemap = np.expand_dims(self.sensemap,0)
    self.sensemap_data = self.sensemap
    Y_data = y_im
    self.Y_Y = self.recon(Y_data, self.sensemap_data, np.ones((1,256,256)))
    self.base_reward = self.reward(self.Y_Y)
    self.prev_reward = self.reward(self.Y_Y)/self.base_reward

    self.game_over = False

    del Y_data

    return np.zeros((256,256))

  def step(self, action):
    # If we have already sampeld the line, strongly penalize it
    already_sampled = False
    if self.sampled_lines[action] == 1:
        already_sampled = True

    self.sampled_lines[action] += 1.
    self.mask[action,:] = 1.

    x_kspace = fft2c(self.im0)
    x_kspace = x_kspace*self.mask
    x_im = ifft2c(x_kspace)
    x_im = np.stack((x_im.real, x_im.imag),axis=-1).astype(np.float32)
    #x_im = np.expand_dims(x_im,0)

    # grace
    mag_input = np.abs(self.im0)
    mag_input = np.expand_dims(mag_input, axis=0)
    mag_input = torch.Tensor(mag_input)

    torchvision.utils.save_image(mag_input, "./real_mag_input.png") # comment to change reward
    # endgrace

    X_data = x_im

    self.mask_data = np.expand_dims(self.mask,0)

    self.curr_img = self.recon(X_data, self.sensemap_data, self.mask_data)
    curr_reward = self.reward(self.curr_img-self.Y_Y)/self.base_reward
    #print(curr_reward.item(), action-128)

    # grace
    real = self.curr_img[:, :, 0]
    imag = self.curr_img[:, :, 1]
    output = np.abs(real + 1j * imag)
    output = np.expand_dims(output, axis=0)
    output = torch.Tensor(output)

    torchvision.utils.save_image(output, "./real_output.png") # comment to change reward
    ## end grace

    reward = self.prev_reward - curr_reward
    #reward = reward - 0.02
    self.prev_reward = curr_reward

    if already_sampled:
        reward = -10

    #self.reward_history[int(np.sum(self.sampled_lines))-1] = reward #qqq
    self.line_order[int(np.sum(self.sampled_lines))-1] = action
    self.reward_order[int(np.sum(self.sampled_lines))-1] = reward

    done = False
    #if curr_reward < 0.30:
    #    done = True
    if np.sum(self.sampled_lines) == 256:
        done = True
        self.game_over = True
        

    if done: #qqq
        self.total_done += 1
        np.save('/home_local/grace/og/dopamine/dopamine/recon_env/grace_l2_order.npy',self.line_order.astype(int))
        np.save('/home_local/grace/og/dopamine/dopamine/recon_env/grace_l2_reward.npy',self.reward_order)
        im_out = np.zeros((256,256))
        im_out[np.arange(256),self.line_order.astype(int)] = 1
        im_out = np.expand_dims(im_out,axis=0)
        im_out = torch.Tensor(im_out)
        torchvision.utils.save_image(im_out, "./decision_order.png")
        del im_out
        sys.exit(0)

    #    plt.figure(), plt.plot(np.cumsum(self.reward_history))
    #    plt.figure(), plt.scatter(np.linspace(0,255,256),self.line_order), plt.show()
    #    #np.save('reward_unr_di.npy',self.reward_history)

    info = self.sampled_lines
    new_state = self.curr_img
    new_state = np.abs(new_state[:,:,0] + 1j*new_state[:,:,1])

    del X_data

    return new_state, reward, done, info

  def get_sampled_lines(self):
      return self.sampled_lines

  def get_curr_img(self):
      return self.curr_img

  def get_ref_img(self):
      return self.Y_Y

  def get_mask(self):
      return self.mask_data

  def get_curr_reward(self):
      return self.prev_reward

  def render(self):
    plt.subplot(1,2,1)
    plt.imshow(np.abs(self.curr_img),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(np.abs(self.im0),cmap='gray')
    plt.show()
    
