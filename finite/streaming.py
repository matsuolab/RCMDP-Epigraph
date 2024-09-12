import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
from functools import partial
from .rcmdp import RCMDP, compute_policy_Q

# This environment is based on [DOPE: Doubly Optimistic and Pessimistic Exploration for Safe Reinforcement Learning](https://arxiv.org/abs/2112.00885).

B = 10  # buffer size
S, A = B+1, 2  # state and action space sizes
N = 1  # number of constraints
DISCOUNT = 0.99 
ITER_LENGTH = 1000  # iteration length for experiment
NUM_SEEDS = 1  # number of evaluation seeds
FIGNAME = "finite/streaming-env"


# +1 objective cost when the buffer size is 0
costs = jnp.zeros((N+1, S, A))
costs = costs.at[0, 0, :].set(1.0)
# +1 constraint cost when taking a fast transmission
costs = costs.at[1, :, 0].set(1.0)
assert costs.shape == (N+1, S, A)

# initial state is s == 0
init_dist = jnp.zeros((S,))
init_dist = init_dist.at[0].set(1.0)
np.testing.assert_allclose(init_dist.sum(axis=-1), 1, atol=1e-6)

threshes = jnp.ones(N) * 0.05 / (1 - DISCOUNT)

S_set = jnp.arange(S)
A_set = jnp.arange(A)


def create_P(mu1: float, mu2: float, rho: float):
    # Given mu1, mu2, rho, create P

    # transition matrix
    P = jnp.zeros((S, A, S))

    # when s==0
    P = P.at[0, 0, 0].set(rho * mu1 + (1-rho) * (1-mu1) + rho * (1 - mu1))
    P = P.at[0, 0, 1].set((1 - rho) * mu1)
    P = P.at[0, 1, 0].set(rho * mu2 + (1-rho) * (1-mu2) + rho * (1 - mu2))
    P = P.at[0, 1, 1].set((1 - rho) * mu2)

    # when 0 < s < N
    for s in range(1, S):
        P = P.at[s, 0, s-1].set(rho * (1 - mu1))
        P = P.at[s, 0, s].set(rho * mu1 + (1-rho) * (1-mu1))
        P = P.at[s, 0, s+1].set((1 - rho) * mu1)
        P = P.at[s, 1, s-1].set(rho * (1 - mu2))
        P = P.at[s, 1, s].set(rho * mu2 + (1-rho) * (1-mu2))
        P = P.at[s, 1, s+1].set((1 - rho) * mu2)

    # when s==B
    P = P.at[B, 0, B].set(rho * mu1 + (1-rho) * (1-mu1) + (1 - rho) * mu1)
    P = P.at[B, 0, B-1].set(rho * (1 - mu1))
    P = P.at[B, 1, B].set(rho * mu2 + (1-rho) * (1-mu2) + (1 - rho) * mu2)
    P = P.at[B, 1, B-1].set(rho * (1 - mu2))

    return P


MU1 = 0.7  # nominal fast transmission rate
MU2 = 1 - MU1  # nominal late transmission rate
RHO = 0.2  # nominal probability of packet leaving
URES = 3  # uncertainty resolution
USIZE = URES ** 3

P = create_P(MU1, MU2, RHO)
np.testing.assert_allclose(P.sum(axis=-1), 1, atol=1e-6)


def create_rcmdp(seed: int):
    S_set = jnp.arange(S)
    A_set = jnp.arange(A)

    # consider MU1 ± eps, MU2 ± eps, RHO ± eps
    eps = 0.1

    # create uncertainty set
    U = jnp.zeros((URES ** 3, S, A, S))
    i = 0
    for mu1 in jnp.linspace(MU1 - eps / 2, MU1 + eps / 2, URES):
        for mu2 in jnp.linspace(MU2 - eps / 2, MU2 + eps / 2, URES):
            for rho in jnp.linspace(RHO - eps / 2, RHO + eps / 2, URES):
                P = create_P(mu1, mu2, rho)
                P = P.reshape(S, A, S)
                U = U.at[i].set(P)
                i = i + 1

    U = jnp.array(U)
    rcmdp = RCMDP(S_set, A_set, DISCOUNT, costs, threshes, U, init_dist)
    return rcmdp