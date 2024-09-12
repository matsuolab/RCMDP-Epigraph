import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from functools import partial
import chex


class RCMDP(NamedTuple):
    S_set: jnp.array  # state space
    A_set: jnp.array  # action space
    discount: float  # discount
    costs: jnp.array  # cost functions
    threshes: jnp.array  # constraint thresholds
    U: jnp.array  # uncertainty set
    init_dist: jnp.array  # initial distribution

    @property
    def S(self) -> int:  # state space size
        return len(self.S_set)

    @property
    def A(self) -> int:  # action space size
        return len(self.A_set)


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