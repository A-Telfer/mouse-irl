from gymnasium.envs.registration import register

register(
     id="mouse_rl/OpenFieldEnv-v0",
     entry_point="mouse_rl.envs:OpenFieldEnv",
     max_episode_steps=300,
)