import gymnasium as gym
from gymnasium import spaces
import subprocess
from pathlib import Path
import numpy as np
import json

# max_steps: internal reasoning steps allowed. Make this high outside debugger
class TerminalBenchEnv(gym.Env):
    def __init__(self, max_steps = 50, verbose=False):
        super().__init__()

        # configure agent
        self.verbose = verbose
        self.config = {}
        self.config["max_steps"] = max_steps # keep max steps small to intervene with guided prompts earlier
        self.config["system_prompt"] = """ You're a problem solver. If prompted for them, write terminal commands in between command tags <cmd> ... </cmd>. Any patches should be in this format: <cmd> echo "[modified_code]" > [filename] </cmd> """
        self.config["messages"] = []
        self.max_outer_steps = 40 # number of guided prompts
        self.harbor_call_id = 0
        self.trial_id = 0
        self.total_step_count = 0

        # Action space. LLM strategy
        self.actions = {
            0: "Inspect the repo structure",
            1: "Read test and validation files, then formulate a first step",
            2: "Debug the latest output", # debug the code, run diagnostics
            3: """Propose a code modification. Modifications to files should be in the format <cmd> echo "[modified_code]" > [filename] </cmd>" """, # patch code
            4: "Install packages that might help pass validation",
            5: "Check if modifications improved the performance"
        }
        self.action_space = spaces.Discrete(len(self.actions))

        # Observations. Compact debugging.
        # [solved flag, last action taken, number of prompts, number of steps including thinking]
        self.observation_space = spaces.Box(
            low=0,
            high=1, #200
            shape=(4,),
            dtype=float,
        )

        self.job_name = f"test_trial{self.trial_id}_call{self.harbor_call_id}"
        self.config["job_name"] = self.job_name 
        self.job_dir = "jobs/"
        self.result_dir = f"{self.job_dir}/{self.job_name}/"
        self.reward = 0
        self.success = False
        self.truncated = False
        self.last_action = 0
        self.write_config()

    def write_config(self):
        Path("agent_config.json").write_text(json.dumps(self.config, indent=2))

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.trial_id += 1
        self.harbor_call_id = 0
        self.config["messages"] = []
        self.reward = 0
        self.success = False
        self.truncated = False
        self.last_action = 0
        self.step_count = 0
        self.job_name = f"test_trial{self.trial_id}_call{self.harbor_call_id}"  
        self.config["job_name"] = self.job_name
        self.job_dir = "jobs/"
        self.result_dir = f"{self.job_dir}/{self.job_name}/" 
        self.total_step_count = 0
        self.write_config()
        return self._obs(), {}

    def step(self, action):

        # update job name and result directory
        self.job_name = f"test_trial{self.trial_id}_call{self.harbor_call_id}"
        self.config["job_name"] = self.job_name 
        self.result_dir = f"{self.job_dir}/{self.job_name}/"
        if self.verbose:
            print(f"Job name: {self.job_name}", flush=True)

        # take step
        # update config
        action = int(action.item()) if hasattr(action, "item") else int(action)
        self.last_action = action
        self.config["messages"].append(self.actions[action])
        self.step_count = len(self.config["messages"])
        self.write_config()

        # execute harbor experiment to validate
        result = subprocess.run(
            ["bash", "custom_launch", self.job_name],
            capture_output=False,
            text=True
        )
        if self.verbose:
            print(f"result.stdout: {result.stdout}")
            print(f"result.stderr:{result}")
        self.harbor_call_id += 1

        # read state
        self.get_result_from_experiment()
        if not self.truncated:
            self.truncated = self.step_count >= self.max_outer_steps
            if self.truncated: 
                print(
                    f"Max Number of prompts reached. Guiding prompts: {self.config["messages"][:5]}...", 
                    flush=True
                )
        
        # print winning policy
        if self.success:
            print(
                f"Success! Reward: {self.reward}. Guiding prompts: {self.config["messages"]}.", 
                flush=True
                )

        observation = self._obs()
        if self.verbose:
            print(f"State: {observation}")
        return observation, self.reward, self.success, self.truncated, {"result": result}


    def get_result_from_experiment(self):

        result_path = Path(self.result_dir + "/result.json")
        if not result_path.exists():
            if self.verbose:
                print(f"Result file not found: {result_path}", flush=True)
            self.reward = 0
            self.success = False
            self.truncated = True
            return

        if self.verbose:
            print(f"result.json found. Reading...", flush=True)
        # read outputs from harbor 
        # get reward, number of steps, any useful outputs
        result = json.loads(result_path.read_text())
        evals = result.get("stats", {}).get("evals", {})
        if not evals:
            self.reward = 0
            self.success = False
            self.truncated = True
            if self.verbose:
                print("evals field is empty", flush=True)
            return
        first_eval = next(iter(evals.values()))
        reward_dict = first_eval.get("reward_stats", {}).get("reward", [])
        if len(reward_dict) == 0:
            self.reward = 0
            self.success = False
            self.truncated = True
            if self.verbose:
                print("reward dict is empty", flush=True)
            return
        self.reward = len(reward_dict.get("1.0", [])) 
        self.success = self.reward > 0

        # read step count
        trajectory_files = list(Path(self.result_dir).rglob(f"{self.job_name}_trajectory.json"))
        if trajectory_files:
            trajectory_path = max(trajectory_files, key=lambda p: p.stat().st_mtime)
            trajectory = json.loads(trajectory_path.read_text())
            total_step_count = len(trajectory)
        else:
            if self.verbose:
                print("No trajectory file found", flush=True)
            self.truncated = True
            self.success = False
            self.reward = 0
            return
        if self.verbose:
            print(
                f"Number of guiding prompts:{len(self.config["messages"])}. Total Steps: {total_step_count}",
                flush=True
                )

        # check if successful
        # shape reward
        self.reward = self.shape_reward(self.reward, total_step_count)
        self.total_step_count = total_step_count
        self.truncated = False

    # observation
    #[solved flag, last action taken, number of prompts, number of steps]
    def _obs(self):
        #return np.array([
        #self.success, 
        #self.last_action, 
        #len(self.config["messages"]),
        #self.total_step_count
        #])
        # normalize
        return np.array([
            float(self.success),
            self.last_action / len(self.actions),
            len(self.config["messages"]) / self.max_outer_steps,
            self.total_step_count / (self.config["max_steps"]*self.max_outer_steps)
            ])

    # reward shaping
    def shape_reward(self, base_reward, total_step_count):
        return base_reward - 0.01 * total_step_count