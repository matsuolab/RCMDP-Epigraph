import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
import chex
from .rcmdp import RCMDP, compute_policy_Q, compute_policy_visit_s, compute_policy_worst_values

# The overall experiments will finish about 30 minutes using 20 CPUs

S, A = 15, 5  # state and action space sizes
N = 1  # number of constraints
USIZE = 5  # size of uncertainty set
DISCOUNT = 0.99 
ITER_LENGTH = 1000  # iteration length for experiment
NUM_SEEDS = 20  # number of evaluation seeds
FIGNAME = "finite/random-env"


@partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0)
@partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=0)
def compute_greedy_Q(discount: float, iter: int, cost: jnp.ndarray, P: jnp.ndarray):
    """Compute a greedy Q function with respect to the constraint cost function in P
    Args:
        discount (float)
        cost (jnp.ndarray)
        P (jnp.ndarray)

    Returns:
        optimal_Q (jnp.ndarray): (SxA)の行列
    """

    def backup(optimal_Q):
        next_v = P @ optimal_Q.min(axis=1)
        assert next_v.shape == (S, A)
        return cost + discount * next_v
    
    optimal_Q = jnp.zeros((S, A))
    body_fn = lambda i, Q: backup(Q)
    return jax.lax.fori_loop(0, iter, body_fn, optimal_Q)


def create_rcmdp(seed: int):
    key = PRNGKey(seed)

    S_set = jnp.arange(S)
    A_set = jnp.arange(A)
    const = jnp.zeros(N)  # dummy

    # randomly create cost function
    costs = jnp.ones((N+1, S, A))
    key, _key = jax.random.split(key)
    zero_mask = jax.random.bernoulli(_key, p=0.5, shape=costs.shape)
    costs = costs * zero_mask
    costs = costs.at[1].set(1 - costs[0])

    # create initial distribution
    key, _key = jax.random.split(key)
    init_dist = jax.random.dirichlet(key=_key, alpha=jnp.array([0.1] * S))
    # np.testing.assert_allclose(init_dist.sum(axis=-1), 1, atol=1e-6)

    # create uncertainty set
    U = jnp.zeros((USIZE, S, A, S))
    for u in range(USIZE):
        key, _key = jax.random.split(key)
        P = jax.random.dirichlet(key=_key, alpha=jnp.array([0.05] * S), shape=((S*A,)))
        P = P.reshape(S, A, S)
        # np.testing.assert_allclose(P.sum(axis=-1), 1, atol=1e-6)
        U = U.at[u].set(P)

    rcmdp = RCMDP(S_set, A_set, DISCOUNT, costs, const, U, init_dist)


    # set the constraint threshold based on the possible constraint satisfaction
    const = 0
    greedy_Qs = compute_greedy_Q(rcmdp.discount, int(1 / (1-rcmdp.discount)) + 100, rcmdp.costs, rcmdp.U)
    greedy_idxes = greedy_Qs.argmin(axis=-1).reshape(-1, S)
    greedy_policy = jnp.zeros((S, A))
    for i in range((N+1) * USIZE):
        greedy_policy = greedy_policy.at[jnp.arange(S), greedy_idxes[i]].add(1)
    greedy_policy = greedy_policy / ((N+1) * USIZE)
    # np.testing.assert_allclose(greedy_policy.sum(axis=-1), 1, atol=1e-6)

    threshes = []
    Qs = compute_policy_Q(rcmdp.discount, greedy_policy, costs, rcmdp.U)
    Vs = (greedy_policy.reshape(1, 1, S, A) * Qs).sum(axis=-1)
    Js = (Vs * rcmdp.init_dist.reshape(1, 1, S)).sum(axis=-1)
    # assert Js.shape == (N + 1, USIZE)
    threshes = Js.max(axis=1)[1:]

    rcmdp = rcmdp._replace(threshes=jnp.array(threshes))
    return rcmdp


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
print("test passed")
