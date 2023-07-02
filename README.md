# demo of bayesian modeling (variational)
The purpose of this project is to demonstrate the implementation and results of the optimization of a probabilistic model. Here the generative model
consists out of a 2D latent variable that encodes for a 784 image vector. Gaussian distributions are used for the prior and conditional distributions.
The model can be trained by using a variational distribution to approximate the posterior or by Monte Carlo importance sampling. The following figures
compare the generative capabilities of the latent space of two such trained models.

![alt text](https://raw.githubusercontent.com/jgwiese/prob_model_variational/main/.msc/vi.png "Generative Capabilities of the latent space after training by VI")
**Figure 1**: After training the model with variational inference almost all images from the training data set are reconstructed by exploring the latent distribution.
<br>
<br>

![alt text](https://raw.githubusercontent.com/jgwiese/prob_model_variational/main/.msc/sampling.png "Generative Capabilities of the latent space after training by Sampling")
**Figure 2**: After training the model with Monte Carlo importance sampling it seems like the generated images have very smooth transitions when exploring the latent space.
However some samples from the training data set appear to be missing.

Training of the model by variational inference is much faster than training by importance sampling, because a simplified version of the ELBO can be derived 
that does not require the evaluation of probability distributions as it is necessary in the case of importance sampling. The generative capabilities of the model by using sampling
however should be only restricted by the amount of samples used during optimization and the number of training iterations.

For basic usage and more functionality demonstration have a look at the 
[jupyter notebook](https://github.com/jgwiese/prob_model_variational/blob/main/image_learning.ipynb).
You could also contact me for details regarding the model and the optimization objective derivation. 
