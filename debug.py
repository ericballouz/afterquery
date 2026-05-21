from stable_baselines3 import PPO
from env import TerminalBenchEnv

# initialize
env = TerminalBenchEnv(verbose=True)

# the optimizer takes in the environment1
model = PPO(
    "MlpPolicy",
    env,
    n_steps=5,
    batch_size=5,
    n_epochs=1,
    verbose=1,
)
model.learn(total_timesteps=2)

obs, info = env.reset()

print("TESTING LEARNED POLICY")
for _ in range(1):
    # model makes an learned guess
    action, _ = model.predict(obs)

    # environment is updated
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)

    print("Reward:", reward)
