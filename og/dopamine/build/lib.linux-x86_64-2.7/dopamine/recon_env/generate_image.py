import val_recon_env as re
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

recon = re.fft_recon()
reward = re.L2_reward()

R = 1
indices = np.load('unrolled_l2_order.npy')
R2 = indices[:int(256//R)]
mask = np.zeros((256,256))
mask[R2,:] = 1.
#plt.figure(), plt.imshow(mask), plt.show()

data_file = '/mnt/data/grace/pytorch_data.h5'
f = h5.File(data_file,'r')
kspace = f['validate_kspace']
sense = f['validate_sense']

idx = 83

complex_kspace = kspace[idx,0:8,:,:] + 1j*kspace[idx,8:,:,:]
complex_kspace = complex_kspace/(np.amax(np.abs(complex_kspace))+1e-6)
complex_kspace = mask*complex_kspace
complex_im = re.ifft2c(complex_kspace)
complex_sense = sense[idx,0:8,:,:] + 1j*sense[idx,8:,:,:]
im0 = complex_im*np.conj(complex_sense)
im0 = np.sum(im0,axis=0)

complex_im = np.stack((im0.real,im0.imag),axis=-1)
sense = np.stack((complex_sense.real,complex_sense.imag),axis=-1)
sense = np.transpose(sense, (0,3,1,2))
mask = np.expand_dims(mask,0)

im = recon(complex_im, sense, mask)
im = im[:,:,0] + 1j*im[:,:,1]
im = np.abs(im)
im = np.rot90(im,-1)

plt.figure()
plt.imshow(im,cmap='gray')
plt.axis('off')
plt.show()
