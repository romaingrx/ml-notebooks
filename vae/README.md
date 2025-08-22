# Core problem

> I'm not 100% sure that there are no mistakes in these notations.

We want to learn the distribution of $p(x|z)$ to be able to generate new samples of the data by sampling $z \sim p(z)$ and then $x \sim p(x|z)$.

The main objective is to maximize the log-likelihood of our data, basically learning the distribution of the data, i.e. 

$$\text{maximize} \log{p(x)} = \log{\int{p(x|z) p(z) \delta{z}}}$$

Problem, the integral is **intractable**.

We don't have access to $p(z|x)$ so we'll approximate it with a NN $q_{\theta}(z|x)$ and derive a new a new objective:


$$
\begin{align*}
\log{p(x)} &= \log{\int{p(x|z) p(z) \delta{z}}} \\
&= \log{\int{p(x|z) p(z) \dfrac{q_{\theta}(z|x)}{q_{\theta}(z|x)} \delta{z}}} \\
&= \log{\int{q_{\theta}(z|x) \dfrac{p(x|z) p(z)}{q_{\theta}(z|x)} \delta{z}}} \\
&= \log{\mathbb{E}_{z \sim q_{\theta}(z|x)} \left[ \dfrac{p(x|z) p(z)}{q_{\theta}(z|x)} \right]} \quad \text{(becomes an expectation over } q(z|x)) \\
&\geq \mathbb{E}_{z \sim q_{\theta}(z|x)} \left[ \log{ \dfrac{p(x|z) p(z)}{q_{\theta}(z|x)}} \right] \quad \text{(using Jensen's inequality since log is concave)} \\
&= \mathbb{E}_{z \sim q_{\theta}(z|x)} \left[ \log{p(x|z)} + \log{p(z) - \log{q_{\theta}(z|x)}} \right] \\
&= \underbrace{\mathbb{E}_{z \sim q_{\theta}(z|x)} \left[ \log{p(x|z)} \right]}_{\text{Reconstruction term}} - \underbrace{\text{KL}(q_{\theta}(z|x) \parallel p(z))}_{\text{Regularization term}} \\
\end{align*}
$$


So with the new objective in place, we can derive the loss function for our dataset $D = \{x_1, x_2, \dots, x_n\}$

$$ L = \dfrac{1}{N} \sum_{x_i \in D} [E_{z \sim q(z|x_i)}[\log{p(x_i|z)}] - KL(q(z|x_i) || p(z))] $$

The only issue is that we can't backpropagate through the sampling step, so we need to use the reparametrization trick and use an external source of randomness. Instead of sampling $z$ directly, we'll sample a noise $\epsilon$ from a standard normal distribution $\epsilon \sim \mathcal{N}(0, 1)$ and then use the following transformation:

$$ z = g_{\theta}(x, \epsilon) = \mu_{\theta}(x) + \sigma_{\theta}(x) \odot \epsilon $$

where $\mu_{\theta}(x)$ and $\sigma_{\theta}(x)$ are the mean and standard deviation of the latent space, which are also learned by the network. This way, we can backpropagate through the sampling step and we obtain the following loss function:

$$ L = \dfrac{1}{N} \sum_{x_i \in D} [E_{\epsilon \sim \mathcal{N}(0, 1)}[\log{p(x_i|g_{\theta}(x_i, \epsilon))}] - KL(q(z|x_i) || p(z))] $$
