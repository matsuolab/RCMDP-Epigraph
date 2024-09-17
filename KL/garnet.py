import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from functools import partial
import chex
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative
from .rcmdp import RCMDP

# The overall experiments will finish about 30 minutes using 20 CPUs

S, A = 5, 3  # state and action space sizes
DIRICHLET = 0.1
N = 4  # number of constraints
KL_PEN = 1.0
DISCOUNT = 0.99
ITER_LENGTH = 1000  # iteration length for experiment
NUM_SEEDS = 10  # number of evaluation seeds
FIGNAME = f"KL/garnet-env-{S}-{A}-{DIRICHLET}-{KL_PEN}-{DISCOUNT}"
DP_ITER = int(1 / (1 - DISCOUNT)) * 2


# >>>>> KL uncertainty set >>>>>

min_eps = jnp.finfo(jnp.float64).resolution

@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def compute_worst_P(Psa: jnp.ndarray, V: jnp.ndarray):
    nV = V.max() - V  # worst_PV_loss is for reward function. Taking negative for cost.
    return jax.nn.softmax(jnp.log(Psa) - nV / KL_PEN)


# <<<<< KL uncertainty set <<<<<


@jax.jit
def compute_policy_matrix(policy: jnp.ndarray):
    """
    Args:
        policy (jnp.ndarray): (SxA) array

    Returns:
        policy_matrix (jnp.ndarray): (SxSA) array
    """
    S, A = policy.shape
    PI = policy.reshape(1, S, A)
    PI = jnp.tile(PI, (S, 1, 1))
    eyes = jnp.eye(S).reshape(S, S, 1)
    PI = (eyes * PI).reshape(S, S*A)
    return PI


@partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0)
def compute_robust_policy_Q(discount: float, policy: jnp.ndarray, cost: jnp.ndarray, nominal_P: jnp.ndarray):
    """ Do robust_policy evaluation with cost and transition kernel
    Args:
        discount (float): discount factor
        policy (jnp.ndarray): (SxA) array
        cost (jnp.ndarray): cost function. (SxA) array
        nominal_P (jnp.ndarray): transition kernel. (SxAxS) array

    Returns:
        Q (jnp.ndarray): (SxA) array
    """
    S, A = policy.shape
    chex.assert_shape(cost, (S, A))
    chex.assert_shape(nominal_P, (S, A, S))

    def condition_fn(loop_args):
        k, pQ, Q = loop_args
        return (jnp.abs(pQ - Q).max() > 1e-6) & (k < DP_ITER)

    def loop_fn(loop_args):
        k, pQ, Q = loop_args
        V = (policy * Q).sum(axis=-1)
        worst_P = compute_worst_P(nominal_P.reshape(S * A, S), V)
        pQ = Q
        Q = cost + discount * worst_P.reshape(S, A, S) @ V
        k = k + 1
        return k, pQ, Q
   
    pQ = jnp.ones((S, A)) * jnp.inf
    Q = jnp.zeros((S, A))
    _, _, Q = jax.lax.while_loop(condition_fn, loop_fn, (0, pQ, Q))
    return Q


@partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=0)
def compute_policy_visit_s(discount: float, policy: jnp.ndarray, init_dist: jnp.ndarray, P: jnp.ndarray):
    """ Compute (unnormalized) occupancy measure of a policy
    Args:
        discount (float): discount factor
        policy (jnp.ndarray): (SxA) array
        init_dist: initial distribution
        P (jnp.ndarray): (SxAxS) array

    Returns:
        d_pi (jnp.ndarray): (S) array
    """
    S, A = policy.shape
    Pi = compute_policy_matrix(policy)
    PiP = Pi @ P.reshape(S*A, S) 
    d_pi = init_dist @ jnp.linalg.inv(jnp.eye(S) - discount * PiP)
    return d_pi


@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
def compute_worst_P_costs(P: jnp.ndarray, V: jnp.ndarray):
    S, A, S = P.shape
    P = P.reshape(-1, S)
    return compute_worst_P(P, V).reshape(S, A, S)


# @jax.jit
def compute_policy_worst_values(policy: jnp.ndarray, rcmdp: RCMDP):
    """
    Args:
        policy (jnp.ndarray)
        rcmdp (RCMDP)

    Returns:
        worst_P_Q (jnp.ndarray): (N+1) x S x A. Values of Q_{n,U}(s, a)
        worst_P_occ (jnp.ndarray): (N+1) x S. Values of d_{n, U}(s)
        worst_P_J (jnp.ndarray): (N+1) vector. Values of J_{n, U}
    """
    S, A = policy.shape
    Np, *_ = rcmdp.costs.shape

    rob_Qs = compute_robust_policy_Q(rcmdp.discount, policy, rcmdp.costs, rcmdp.nominal_P)  # N+1 x |U| x S x A
    chex.assert_shape(rob_Qs, (Np, S, A))
    rob_Vs = (rob_Qs * policy.reshape(1, S, A)).sum(axis=-1)
    worst_P_J = jnp.sum(rob_Vs * rcmdp.init_dist.reshape(1, S), axis=-1)
    chex.assert_shape(worst_P_J, (Np, ))

    worst_Ps = compute_worst_P_costs(rcmdp.nominal_P, rob_Vs)
    chex.assert_shape(worst_Ps, (Np, S, A, S))

    worst_P_occ = compute_policy_visit_s(rcmdp.discount, policy, rcmdp.init_dist, worst_Ps)

    chex.assert_shape(worst_P_occ, (Np, S))
    return rob_Qs, worst_P_occ, worst_P_J


def create_rcmdp(seed: int):
    key = PRNGKey(seed)

    S_set = jnp.arange(S)
    A_set = jnp.arange(A)
    const = jnp.zeros(N)  # dummy

    # randomly create cost function
    costs = jnp.ones((N+1, S, A))
    key, _key = jax.random.split(key)
    zero_mask = jax.random.bernoulli(_key, p=0.9, shape=costs.shape)
    costs = costs * zero_mask

    # create initial distribution
    key, _key = jax.random.split(key)
    init_dist = jax.random.dirichlet(key=_key, alpha=jnp.array([0.5] * S))
    # np.testing.assert_allclose(init_dist.sum(axis=-1), 1, atol=1e-6)

    # create nominal transition function
    key, _key = jax.random.split(key)
    nominal_P = jax.random.dirichlet(key=_key, alpha=jnp.array([DIRICHLET] * S), shape=((S*A,)))
    nominal_P = nominal_P.reshape(S, A, S)
    # np.testing.assert_allclose(nominal_P.sum(axis=-1), 1, atol=1e-6)

    rcmdp = RCMDP(S_set, A_set, DISCOUNT, costs, const, nominal_P, init_dist)

    threshes = []
    key, _key = jax.random.split(key)
    random_policy = jax.random.dirichlet(key=_key, alpha=jnp.array([0.2] * A), shape=((S,)))
    Qs = compute_robust_policy_Q(rcmdp.discount, random_policy, costs, rcmdp.nominal_P)
    Vs = (random_policy.reshape(1, S, A) * Qs).sum(axis=-1)
    Js = (Vs * rcmdp.init_dist.reshape(1, S)).sum(axis=-1)
    # assert Js.shape == (N + 1, USIZE)
    threshes = Js[1:]

    rcmdp = rcmdp._replace(threshes=jnp.array(threshes))
    return rcmdp


# ===== test =====

rcmdp = create_rcmdp(0)
policy = jnp.ones((S, A)) / A
Qs = compute_robust_policy_Q(rcmdp.discount, policy, rcmdp.costs, rcmdp.nominal_P)
assert not jnp.any(jnp.isnan(Qs))
Vs = (Qs * policy.reshape(1, S, A)).sum(axis=-1)
Js = jnp.sum(Vs * rcmdp.init_dist.reshape(1, S), axis=-1)

chex.assert_shape(Qs, (N+1, S, A))
chex.assert_shape(Js, (N+1, ))

worst_P_Q, worst_P_occ, worst_P_J = compute_policy_worst_values(policy, rcmdp)
print("test passed")
