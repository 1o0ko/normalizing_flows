# Variational Inference with Normalizing Flows

## Types of Generative Models


1. **Generative Aversarial Networks**:  Generator and Discriminator, discriminator learns to distinguish the real data from the fake samples that are produced by the generator model. 
2. **Variational Autoencoders**: VAE inexplicitly optimizes the log-likelyhood of the data by maximizing the evidence lower bound (ELBO)
3. **Flow-based** generative models: are constructed by a sequence of invertible transformations. Unlike GANs and VAEs the model explicitly learns the true data distribution $p(\mathbf x)$ and the loss function is simply the negative log-likelyhood.


Stolen from From [Lilian Weng's](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#jacobian-matrix-and-determinant) blog:


![alt text](https://lilianweng.github.io/lil-log/assets/images/three-generative-models.png)
