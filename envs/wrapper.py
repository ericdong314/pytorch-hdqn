# This wrapper is copied from jumanji module with modified reset() and step() that return jumanji_state

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import chex
import dm_env.specs
import gym
import jax
import jax.numpy as jnp
import numpy as np

from jumanji import specs, tree_utils
from jumanji.env import Environment, State
from jumanji.types import TimeStep

Observation = TypeVar("Observation")

# Type alias that corresponds to ObsType in the Gym API
GymObservation = Any

class JumanjiToGymWrapper(gym.Env):
    """A wrapper that converts a Jumanji `Environment` to one that follows the `gym.Env` API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: Environment, seed: int = 0, backend: Optional[str] = None):
        """Create the Gym environment.

        Args:
            env: `Environment` to wrap to a `gym.Env`.
            seed: the seed that is used to initialize the environment's PRNG.
            backend: the XLA backend.
        """
        self._env = env
        self.metadata: Dict[str, str] = {}
        self._key = jax.random.PRNGKey(seed)
        self.backend = backend
        self._state = None
        self.observation_space = specs.jumanji_specs_to_gym_spaces(
            self._env.observation_spec()
        )
        self.action_space = specs.jumanji_specs_to_gym_spaces(self._env.action_spec())

        def reset(key: chex.PRNGKey) -> Tuple[State, Observation, Optional[Dict]]:
            """Reset function of a Jumanji environment to be jitted."""
            state, timestep = self._env.reset(key)
            return state, timestep.observation, timestep.extras

        self._reset = jax.jit(reset, backend=self.backend)

        def step(
            state: State, action: chex.Array
        ) -> Tuple[State, Observation, chex.Array, bool, Optional[Any]]:
            """Step function of a Jumanji environment to be jitted."""
            state, timestep = self._env.step(state, action)
            done = jnp.bool_(timestep.last())
            return state, timestep.observation, timestep.reward, done, timestep.extras

        self._step = jax.jit(step, backend=self.backend)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """Resets the environment to an initial state by starting a new sequence
        and returns the first `Observation` of this sequence.

        Returns:
            obs: an element of the environment's observation_space.
            info (optional): contains supplementary information such as metrics.
        """
        if seed is not None:
            self.seed(seed)
        key, self._key = jax.random.split(self._key)
        self._state, obs, extras = self._reset(key)

        # Convert the observation to a numpy array or a nested dict thereof
        obs = jumanji_to_gym_obs(obs)

        if return_info:
            info = jax.tree_util.tree_map(np.asarray, extras)
            return obs, info
        else:
            return obs, self._state  # type: ignore

    def step(
        self, action: chex.ArrayNumpy
    ):
        """Updates the environment according to the action and returns an `Observation`.

        Args:
            action: A NumPy array representing the action provided by the agent.

        Returns:
            observation: an element of the environment's observation_space.
            reward: the amount of reward returned as a result of taking the action.
            terminated: whether a terminal state is reached.
            info: contains supplementary information such as metrics.
        """

        action = jnp.array(action)  # Convert input numpy array to JAX array
        self._state, obs, reward, done, extras = self._step(self._state, action)

        # Convert to get the correct signature
        obs = jumanji_to_gym_obs(obs)
        reward = float(reward)
        terminated = bool(done)
        info = jax.tree_util.tree_map(np.asarray, extras)

        return obs, reward, terminated, info, self._state

    def seed(self, seed: int = 0) -> None:
        """Function which sets the seed for the environment's random number generator(s).

        Args:
            seed: the seed value for the random number generator(s).
        """
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human") -> Any:
        """Renders the environment.

        Args:
            mode: currently not used since Jumanji does not currently support modes.
        """
        del mode
        return self._env.render(self._state)

    def close(self) -> None:
        """Closes the environment, important for rendering where pygame is imported."""
        self._env.close()

    @property
    def unwrapped(self) -> Environment:
        return self._env


def jumanji_to_gym_obs(observation: Observation) -> GymObservation:
    """Convert a Jumanji observation into a gym observation.

    Args:
        observation: JAX pytree with (possibly nested) containers that
            either have the `__dict__` or `_asdict` methods implemented.

    Returns:
        Numpy array or nested dictionary of numpy arrays.
    """
    if isinstance(observation, jnp.ndarray):
        return np.asarray(observation)
    elif hasattr(observation, "__dict__"):
        # Applies to various containers including `chex.dataclass`
        return {
            key: jumanji_to_gym_obs(value) for key, value in vars(observation).items()
        }
    elif hasattr(observation, "_asdict"):
        # Applies to `NamedTuple` container.
        return {
            key: jumanji_to_gym_obs(value)
            for key, value in observation._asdict().items()  # type: ignore
        }
    else:
        raise NotImplementedError(
            "Conversion only implemented for JAX pytrees with (possibly nested) containers "
            "that either have the `__dict__` or `_asdict` methods implemented."
        )
