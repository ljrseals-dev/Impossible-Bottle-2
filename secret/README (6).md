# Bayesian Bare-Metal Image Generation (pure NumPy)
Run: `python run_all.py` (needs only numpy).

| File | Method | Bayesian element |
|---|---|---|
| ddpm.py | Diffusion | Exact Gaussian posterior q(x_{t-1}|x_t,x0); accumulated predictive variance |
| vae.py | VAE | ELBO = likelihood - KL(q(z|x)||p(z)); multi-z uncertainty map |
| pixel_ar.py | Autoregressive | Laplace-posterior Bayesian logistic conditionals; entropy abstain flags |
| gibbs_mrf.py | Geman&Geman 1984 | Gibbs sampling + annealing; posterior-mean confidence |
| rbm.py | Boltzmann lineage | CD-k, Gaussian weight prior, free-energy novelty score, multi-chain std |
| bayes_attr_net.py | Pearl-style BN | BIC structure learning, Dirichlet CPTs, exact inference |
| common.py | Shared | ECE calibration, entropy, hash-chained provenance |

Honest limits: toy 8x8 data; models are small and meant to show the math, not match modern quality.
Uncertainty here is approximate (Laplace, MC samples, residual variance), not guaranteed-calibrated;
check ECE and held-out data before trusting outputs. Swap make_toy_data for real images to scale.
