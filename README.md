# Variational Inference with Normalizing Flows

## Types of Generative Models


1. **Generative Aversarial Networks**:  Generator and Discriminator, discriminator learns to distinguish the real data from the fake samples that are produced by the generator model. 
2. **Variational Autoencoders**: VAE inexplicitly optimizes the log-likelyhood of the data by maximizing the evidence lower bound (ELBO)
3. **Flow-based** generative models: are constructed by a sequence of invertible transformations. Unlike GANs and VAEs the model explicitly learns the true data distribution $p(\mathbf x)$ and the loss function is simply the negative log-likelyhood.


Stolen from From [Lilian Weng's](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#jacobian-matrix-and-determinant) blog:


![alt text](https://lilianweng.github.io/lil-log/assets/images/three-generative-models.png)

##  Normalizing Flows


The basic rule for transformation of densities consideres an invertible, smooth mapping $f: \mathbb{R}^D \rightarrow  \mathbb{R}^D$ with an inverse $f^{-1}=g$, such that  $ g \circ f (\textbf{z}) = \textbf{z}$.  If we use this mapping to transform a random variable $\mathbf{z}$ with distribution $q(\mathbf{z})$, then the resulting random variable $\mathbf{z}^\prime = f(\mathbf{z}$) has a distribution:

$$
q(\mathbf{z}^{\prime}) = q(\mathbf{z}) 
  \left| 
    \det \frac{\partial f^{-1}}{\partial \mathbf{z}^\prime}
  \right| = 
   q(\mathbf{z}) 
  \left| 
    \det \frac{\partial f}{\partial \mathbf{z}}
  \right|^{-1},
$$
where the last equality can be obtained by applying [the inverse function theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem) and taking advantage of the property of Jacobians of intertible functions.

The density $q_K(\mathbf z)$ obtained by successively transforming a random variable $\mathbf z_0$ with distribution
$q_0$ through a chain of $K$ transformations $f_k$ is:

\begin{align}
  \mathbf z_K &= f_K \circ \ldots \circ f_1( \mathbf z_0), \\\
  \ln q_K (\mathbf z_K) &= \ln q_0(\mathbf z_0) - \sum_{k=1}^{K} \ln \det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}}.
\end{align}

The formalism of normalizing flows now gives us a systematic
way of specifying the approximate posterior distributions
$q(\mathbf z| \mathbf x)$ required for variational inference. With an
appropriate choice of transformations $f_K$, we can initially
use simple factorized distributions such as an independent
Gaussian, and apply normalizing flows of different lengths
to obtain increasingly complex and multi-modal distributions.



From [Lilian Weng's  blog](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#jacobian-matrix-and-determinant) (she uses $p_i$ instead of $q_i$):
![alt text](https://lilianweng.github.io/lil-log/assets/images/normalizing-flow.png)

###  Remark:
If  $\mathbf{p}$ is a point in $\mathbb{R}^D$ and $f$ is differentiable at $\mathbf{p}$, then its derivative is given by $J_f(\mathbf{p})$. In this case, the linear map described by $J_f(\mathbf{p})$ is the best linear approximation of $f$ near the point $\mathbf{p}$, in the sense that

$$
\mathbf f(\mathbf x) = \mathbf f(\mathbf p) + \mathbf J_{\mathbf f}(\mathbf p)(\mathbf x - \mathbf p) + o(\|\mathbf x - \mathbf p\|),
$$

where $\mathbf x$ is close to $\mathbf p$ and where $o$ is the little o-notation.

Since, we can percieve the Jacobian of $f: \mathbb{R}^D \rightarrow  \mathbb{R}^D$ as locally linear map, we can describe the space distortions using the determinant: geometrically the absolute value of the Jacobian determinant gives the magnification/scalling factor when we transform an area or volume. It intuitevely make sense, that if function changes the volume by $a$ it's inverse should change the volme by $\frac{1}{a}$. 
