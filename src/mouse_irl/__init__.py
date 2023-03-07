from gymnasium.envs.registration import register
from . import datasets

register(
     id="mouse_irl/OpenFieldEnv-v0",
     entry_point="mouse_irl.envs:OpenFieldEnv",
     max_episode_steps=300,
)