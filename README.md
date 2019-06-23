# Hand-on With Generative Adverserial Networks

In this repo, I explore the Generative Adverserial Networks (GAN). A well trained
GAN lets you draw samples from a general distribution. For example, you can create
bedroom images or could create digits images from a random input.
This is akin to drawing random numbers for example from a Gaussian distribution.
These kind of networks
are in contrast to more widely known predictive networks which lets you distinguish
between a cat image and a human image for example.

GANs where brought into limelight by Ian Goodfellow *et al.* in their [2014 paper](https://arxiv.org/pdf/1406.2661.pdf). I have tried to recreate this paper in
the script [1_generative_mnist.py](1_generative_mnist.py). A simple GAN involves
2 separate neural networks. One called the generative-network and another, the discriminative-network.
A generative-network takes random vector as input and gives out an image (generated image)
as output. The discriminative-network takes an image as input and gives the verdict
if the image was a real image (follows the underlying complex distribution) or a fake image (not belonging to the
underlying complex distribution). I highly recommend you read [this blog post] (https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3)
 for a simplistic overview of the GAN before reading the paper from Goodfellow.



## Scripts
### `1_generative_mnist.py`
This script learns the underlying distribution of mnist dataset. After training is
complete it draws some images from this distribution. Here is how those images look like
after training for 400 epochs.

### try transposed conv for generator, 


## Resources
- [A simplistic explaination of GAN](https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3)
- [NIPS2016 GAN tutorial by Ian Goodfellow](https://www.youtube.com/watch?v=AJVyzd0rqdc)
- [2014-2019 GAN literature review](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/)
- [Some practical tips on improving GAN performances](https://github.com/soumith/ganhacks)
