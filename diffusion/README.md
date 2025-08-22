# Resemblance to the VAE problem

The core problem is very similar to [the VAE one](../vae/README.md), simply in this case, rather than learning $q_\theta(z|x)$, the forward process is fixed with


$$q(x_{1:T}|x_0) = \prod_{t=1}^{T}{q(x_t|x_{t-1})}$$

# Forward process

And we want to converge to a standard normal distribution and learn to predict the noise added to a noisy sample. So

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

with $\beta_t \in [0, 1]$ and scheduled to be a function of $t$.


# Main objective

The main objective is to learn the reverse process $p_\theta(x_{t-1}|x_t)$ to be able to generate new samples of the data by sampling $x_t \sim p(x_t)$ and then $x_{t-1} \sim p_\theta(x_{t-1}|x_t)$.

We still maximize the log-likelihood of our data, i.e.

$$ 
\begin{align*}
    \log{p(x_0)} &= \log{\int{p(x_{0:T}) \dfrac{q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} \delta{x_{1:T}}}} \\
    &= \log{\int{q(x_{1:T}|x_0) \dfrac{p(x_{0:T})}{q(x_{1:T}|x_0)} \delta{x_{1:T}}}} \\
    &= \log{\mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \dfrac{p(x_{0:T})}{q(x_{1:T}|x_0)} \right]} \\
    &\ge \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)} \left[ \dfrac{p(x_{0:T})}{q(x_{1:T}|x_0)} \right] \\
\end{align*}
$$

The loss becomes:


$$L = \mathbb{E}_q\left[\underbrace{D_{KL}(q(x_T|x_0)||p(x_T))}_{\text{Prior matching}} + \sum_{t>1} \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))}_{\text{Denoising terms}} \underbrace{- \log p_\theta(x_0|x_1)}_{\text{Reconstruction}}\right]$$

# Reparametrization trick

Instead of paramaterizing $\mu_\theta$, we reparameterize using the noise prediction, since $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ and $\bar{\alpha}_t = \prod_{i=1}^{t}{\alpha_i}$ and $\alpha_t = 1 - \beta_t$.

So we can write:

$$\mu_\theta(x_t, t) = \dfrac{1}{\sqrt{\alpha_t}} \left(x_t - \dfrac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

This transforms the loss into:

$$L = \mathbb{E}_{t,x_0,\epsilon}\left[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2\right]$$


> **Beauty behind the loss**
> We just need to learn how much noise was added to the sample at time $t$ to be able to reconstruct it and the loss is just a simple MSE between the predicted noise and the actual noise 🤯