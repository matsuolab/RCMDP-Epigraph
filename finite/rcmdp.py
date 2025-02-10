import jax.numpy as jnp
from typing import NamedTuple


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
