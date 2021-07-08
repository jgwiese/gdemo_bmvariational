# prob_model_variational

The purpose of this project is to demonstrate the implementation and results of a probabilistic model. Here the generative model
consists out of a 2D latent variable that encodes for a 784 image vector. Gaussian distributions are used for the prior and conditional distributions.
The model can be trained by using variational distribution to approximate the posterior or by Monte-Carlo importance sampling. The following figures
compare the generative capabilities of the latent space of two such trained models.

![alt text](https://raw.githubusercontent.com/jgwiese/prob_model_variational/main/.msc/vi.png "Generative Capabilities of the latent space after training by VI")
**Figure 1**: 
<br>
<br>

![alt text](https://raw.githubusercontent.com/jgwiese/prob_model_variational/main/.msc/vi.png "Generative Capabilities of the latent space after training by Sampling")
**Figure 2**: 

For basic usage and more functionality demonstration have a look at the 
[jupyter notebook](https://github.com/jgwiese/prob_model_variational/blob/main/image_learning.ipynb).
You can also contact me for details regarding the model and the optimization objective derivation. 
