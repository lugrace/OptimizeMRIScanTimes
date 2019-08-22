import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('/home/davidyzeng/machinelearning/cones/python/packages/mrirecon')
import fftc

if not os.path.exists('l2_avg.npy'):
    data_file = '/home_local/grace/pytorch_data.h5'
    f = h5.File(data_file,'r')
    kspace = f['train_kspace']
    sense = f['train_sense']

    curr_avg = np.zeros((256,256),dtype='f')

    for idx in range(kspace.shape[0]):
	if idx % 100 == 0:
	    print(idx)
	complex_kspace = kspace[idx,0:8,:,:] + 1j*kspace[idx,8:,:,:]
	complex_sense = sense[idx,0:8,:,:] + 1j*sense[idx,8:,:,:]
	complex_im = fftc.ifft2c(complex_kspace)
	im0 = complex_im*np.conj(complex_sense)
	im0 = np.sum(im0,axis=0)

	curr_kspace = fftc.fft2c(im0)
	curr_kspace = np.abs(curr_kspace)
	curr_kspace = curr_kspace/np.sum(curr_kspace)
	curr_avg = curr_avg + (curr_kspace - curr_avg)/(idx+1)

    print(np.sum(curr_avg))
    np.save('l2_avg.npy',curr_avg)
    plt.figure()
    plt.imshow(curr_avg)
    plt.show()
else:
    curr_avg = np.load('l2_avg.npy')

deep_order = np.load('val.npy')
phase_pdf = np.sum(curr_avg, axis=0)
curr_order = np.argsort(-phase_pdf)

#plt.figure()
#plt.plot(np.arange(256)-128,phase_pdf)
#plt.show()

data_file = '/home_local/grace/pytorch_data.h5'
f = h5.File(data_file,'r')
kspace = f['validate_kspace']
sense = f['validate_sense']

idx = 83
complex_kspace = kspace[idx,0:8,:,:] + 1j*kspace[idx,8:,:,:]
complex_sense = sense[idx,0:8,:,:] + 1j*sense[idx,8:,:,:]
complex_im = fftc.ifft2c(complex_kspace)
im0 = complex_im*np.conj(complex_sense)
im0 = np.sum(im0,axis=0)

curr_kspace = fftc.fft2c(im0)
curr_kspace = np.abs(curr_kspace)
curr_kspace = curr_kspace/np.sum(curr_kspace)
curr_kspace = np.sum(curr_kspace, axis=0)

true_order = np.argsort(-curr_kspace)

deep = curr_kspace[deep_order]
curr = curr_kspace[curr_order]
truth = curr_kspace[true_order]
unrolled = np.load('unrolled_l2_reward.npy')
cs = np.load('cs_l2_reward.npy')


plt.figure()
plt.subplot(1,2,1)
plt.semilogy(truth[1:],label='Ground Truth')
plt.semilogy(deep[1:],label='Prior Only')
plt.semilogy(curr[1:],label='RL: FFT Recon')
plt.semilogy(unrolled[1:],label='RL: Deep Recon')
plt.semilogy(cs[1:],label='RL: CS Recon')
plt.legend()
plt.xlabel('Readout #')
plt.title('L2 Reward')

#plt.figure()
plt.subplot(1,2,2)
plt.plot(np.cumsum(truth),label='Ground Truth')
plt.plot(np.cumsum(deep),label='Prior Only')
plt.plot(np.cumsum(curr),label='RL: FFT Recon')
plt.plot(np.cumsum(unrolled),label='RL: Deep Recon')
plt.plot(np.cumsum(cs),label='RL: CS Recon')
plt.legend()
plt.xlabel('Readout #')
plt.title('Cumulative L2 Reward')
plt.show()

