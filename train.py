from stable_baselines3 import PPO
from env import TerminalBenchEnv

# initialize
env = TerminalBenchEnv()

# the optimizer takes in the environment1
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100) # 2000

obs, info = env.reset()

print("TESTING LEARNED POLICY")
for _ in range(2):
    # model makes an learned guess
    action, _ = model.predict(obs)

    # environment is updated
    obs, reward, terminated, truncated, info = env.step(action)

    print("Reward:", reward)
