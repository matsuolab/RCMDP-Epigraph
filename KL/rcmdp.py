import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from functools import partial
import chex
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative


class RCMDP(NamedTuple):
    S_set: jnp.array  # state space
    A_set: jnp.array  # action space
    discount: float  # discount
    costs: jnp.array  # cost functions
    threshes: jnp.array  # constraint thresholds
    nominal_P: jnp.array  # nominal transition
    KL_rad: float  # KL radius
    init_dist: jnp.array  # initial distribution

    @property
    def S(self) -> int:  # state space size
        return len(self.S_set)

    @property
    def A(self) -> int:  # action space size
        return len(self.A_set)
