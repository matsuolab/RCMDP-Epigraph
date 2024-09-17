import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from functools import partial
import chex
from .rcmdp import RCMDP

# The overall experiments will finish about 30 minutes using 20 CPUs

S, A = 7, 4  # state and action space sizes
N = 5  # number of constraints
DIRICHLET = 0.1
USIZE = 1  # size of uncertainty set
DISCOUNT = 0.99
ITER_LENGTH = 1000  # iteration length for experiment
NUM_SEEDS = 10  # number of evaluation seeds
FIGNAME = f"finite/garnet-env-{S}-{A}-{DIRICHLET}-{USIZE}-{DISCOUNT}"


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
@partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=0)
def compute_policy_Q(discount: float, policy: jnp.ndarray, cost: jnp.ndarray, P: jnp.ndarray):
    """ Do policy evaluation with cost and transition kernel
    Args:
        discount (float): discount factor
        policy (jnp.ndarray): (SxA) array
        cost (jnp.ndarray): cost function. (SxA) array
        P (jnp.ndarray): transition kernel. (SxAxS) array

    Returns:
        Q (jnp.ndarray): (SxA) array
    """
    S, A = policy.shape

    Pi = compute_policy_matrix(policy)
    PPi = P.reshape(S*A, S) @ Pi
    Q = jnp.linalg.inv(jnp.eye(S*A) - discount * PPi) @ cost.reshape(S*A)
    return Q.reshape(S, A)


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


@jax.jit
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
    USIZE, *_ = rcmdp.U.shape

    Qs = compute_policy_Q(rcmdp.discount, policy, rcmdp.costs, rcmdp.U)  # N+1 x |U| x S x A
    Vs = (Qs * policy.reshape(1, 1, S, A)).sum(axis=-1)
    Js = jnp.sum(Vs * rcmdp.init_dist.reshape(1, 1, S), axis=-1)
    occs = compute_policy_visit_s(rcmdp.discount, policy, rcmdp.init_dist, rcmdp.U)
    worst_P_idx = jnp.argmax(Js, axis=-1)

    chex.assert_shape(Qs, (Np, USIZE, S, A))
    chex.assert_shape(Js, (Np, USIZE))
    chex.assert_shape(occs, (USIZE, S))
    chex.assert_shape(worst_P_idx, (Np,))

    worst_P_Q = jnp.zeros((Np, S, A))
    for n in range(Np):
        worst_P_Q = worst_P_Q.at[n].set(Qs[n, worst_P_idx[n]])
    worst_P_occ = occs[worst_P_idx]
    chex.assert_shape(worst_P_occ, (Np, S))

    worst_P_J = Js.max(axis=-1)
    chex.assert_shape(worst_P_J, (Np,))
    return worst_P_Q, worst_P_occ, worst_P_J


def create_rcmdp(seed: int):
    key = PRNGKey(seed)

    S_set = jnp.arange(S)
    A_set = jnp.arange(A)

    # randomly create cost function
    costs = jnp.ones((N+1, S, A))
    key, _key = jax.random.split(key)
    zero_mask = jax.random.bernoulli(_key, p=0.9, shape=costs.shape)
    costs = costs * zero_mask
    # costs = costs.at[1].set(1 - costs[0])

    # create initial distribution
    key, _key = jax.random.split(key)
    init_dist = jax.random.dirichlet(key=_key, alpha=jnp.array([0.5] * S))
    # np.testing.assert_allclose(init_dist.sum(axis=-1), 1, atol=1e-6)

    # create uncertainty set
    U = jnp.zeros((USIZE, S, A, S))
    for u in range(USIZE):
        key, _key = jax.random.split(key)
        P = jax.random.dirichlet(key=_key, alpha=jnp.array([DIRICHLET] * S), shape=((S*A,)))
        P = P.reshape(S, A, S)
        # np.testing.assert_allclose(P.sum(axis=-1), 1, atol=1e-6)
        U = U.at[u].set(P)

    rcmdp = RCMDP(S_set, A_set, DISCOUNT, costs, 0, U, init_dist)

    threshes = []
    key, _key = jax.random.split(key)
    random_policy = jax.random.dirichlet(key=_key, alpha=jnp.array([0.2] * A), shape=((S,)))
    Qs = compute_policy_Q(rcmdp.discount, random_policy, costs, rcmdp.U)
    Vs = (random_policy.reshape(1, 1, S, A) * Qs).sum(axis=-1)
    Js = (Vs * rcmdp.init_dist.reshape(1, 1, S)).sum(axis=-1).max(axis=-1)
    # assert Js.shape == (N + 1, USIZE)
    threshes = Js[1:] * 1.01

    return rcmdp._replace(threshes=threshes)


# ===== test =====

rcmdp = create_rcmdp(0)
policy = jnp.ones((S, A)) / A
Qs = compute_policy_Q(rcmdp.discount, policy, rcmdp.costs, rcmdp.U)  # N+1 x |U| x S x A
Vs = (Qs * policy.reshape(1, 1, S, A)).sum(axis=-1)
Js = jnp.sum(Vs * rcmdp.init_dist.reshape(1, 1, S), axis=-1)
ds = compute_policy_visit_s(rcmdp.discount, policy, rcmdp.init_dist, rcmdp.U)

chex.assert_shape(Qs, (N+1, USIZE, S, A))
chex.assert_shape(Js, (N+1, USIZE))
chex.assert_shape(ds, (USIZE, S))

worst_P_Q, worst_P_occ, worst_P_J = compute_policy_worst_values(policy, rcmdp)

if USIZE == 1:
    assert jnp.all(worst_P_occ[0] == worst_P_occ[1])

print("test passed")
