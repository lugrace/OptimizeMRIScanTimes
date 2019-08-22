# OptimizeMRIScanTimes
One of my projects at Stanford (Summer 2019)

Optimizes MRI scanning times by undersampling the Fourier-domain data. This increases the cost-effectiveness of MRI, allowing hospitals to scan more patients. Moreover, reducing scanning times increases patient cooperation so the images are of a higher quality. To do this, I used Tensorflow and Pytorch with Google Dopamine to train a ResNet CNN to continuously infer the optimal sampling trajectory. From the inferred sampling pattern, we are then able to reconstruct high-quality MRI images with respect to image acquisition methods and image quality metrics.

In particular, the specific image acquisition methods we tested were:
    - Inverse Fourier Transform (FFT - the standard reconstruction method)
    - Unrolled Optimization Network (a pretrained deep learning network)
    - Compressed Sensing 

Similarly, the image quality metrics used were:
    - L2 Reward (Euclidean difference between ground-truth and reconstructed image)
    - Perceptual Reward Function (using feature maps from a pretrained VGG network)
    - Discriminator Reward (deep learning network for proof-of-concept)
    - Visual Reward Function (Adversial/Perceptual Combination (increases both high frequency and textural details))

The model was also tested over several different networks with 3, 5, and 7 2D convolutional layers. Overall, we increased the accuracy of reconstructed images by over 40% after implementing deep-learning-based acquisition and perceptual reward function using a pre-trained VGG network.
