#import anthropic
from openai import OpenAI
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pathlib import Path
import json
import re

BASE_URL="http://localhost:11434/v1"
MAX_CHARS = 4000
MAX_TOKENS = 512

class MyAgent(BaseAgent):

    def __init__(self, model_name=None, logs_dir=None, **kwargs):
        super().__init__(model_name=model_name, logs_dir=logs_dir, **kwargs)
        #self.model_name = model_name
        #self.logs_dir = logs_dir

        config = json.loads(Path("agent_config.json").read_text())
        self.max_steps = config["max_steps"]
        self.system_prompt = config["system_prompt"]
        self.messages = config["messages"]
        self.job_name = config["job_name"]
        self.step_count = 0
        self.thinking_step_count = 0
        #self.context = None

    @staticmethod
    def name() -> str:
        return "my_agent"

    def version(self) -> str | None:
        return "1.0.0"

    def extract_commands(self, text):
        # primary format: <cmd>...</cmd>
        commands = re.findall(r'<cmd>(.*?)</cmd>', text, re.DOTALL)
        if commands:
            return [c.strip() for c in commands]
        return []

        # save trajectory
    def save_trajectory(self, trajectory):
        log_file = Path(self.logs_dir) / f"{self.job_name}_trajectory.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(json.dumps(trajectory, indent=2)) 

    async def setup(self, environment: BaseEnvironment) -> None:
        # any setup commands to run in the container before the task
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext
    ) -> None:

        client = OpenAI(base_url=BASE_URL, api_key="ollama") #anthropic.Anthropic()
        #self.context = context
        trajectory = []
        conversation = [
            {"role": "user", "content": instruction}
        ]

        # go through messages, which are actions from the RL environment
        # execute any code
        for message_RL in self.messages:
            self.step_count += 1
            # update conversation
            conversation.append({
                "role": "user",
                "content": message_RL
            })
            trajectory.append({
                    "step": self.step_count, #len(trajectory) + 1,
                    "user_prompt": message_RL
            })
            self.save_trajectory

            # inner agentic loop (LLM will reason on its own in addition to RL actions)
            for _ in range(self.max_steps): 

                # pass message to LLM
                #response = client.messages.create(
                #    model="claude-sonnet-4-5",
                #    max_tokens=4096,
                #    system=self.system_prompt,
                #    messages=conversation,
                #)

                response = client.chat.completions.create(
                    model="mistral",
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        *conversation
                        ],
                        max_tokens=MAX_TOKENS,
                )

                # extract output and command
                #raw_output = response.content[0].text
                raw_output = response.choices[0].message.content

                # update conversation to include both guidance from RL and system outputs
                # copy output into LLM but only up to max_steps times
                conversation.append({
                    "role": "assistant",
                    "content": raw_output,
                })

                # execute any commands
                commands = self.extract_commands(raw_output)
                # if agent doesn't suggest commands on its own, break and check RL messages
                if not commands:
                    self.step_count += 1
                    # model finished, no command to run
                    trajectory.append({ 
                        "step": self.step_count,
                        "assistant_reasoning": raw_output,
                        "command": None,
                        "stdout": None,
                        "stderr": None,
                    })
                    self.save_trajectory(trajectory)
                    break
    
                command_outputs = []
                for command in commands:
                    self.step_count += 1 # extra step
                    # update trajectory
                    result = await environment.exec(command)
                    trajectory.append({
                    "step": self.step_count, #len(trajectory) + 1,
                    "command": command,
                    "assistant_reasoning": raw_output,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    })
                    self.save_trajectory(trajectory)

                    # keep track of terminal output and make them part of the conversation
                    command_outputs.append(
                        f"$ {command}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                    )

                # truncate if output is too long
                command_outputs = "\n\n".join(command_outputs)
                if len(command_outputs) > MAX_CHARS: 
                    command_outputs = command_outputs[-MAX_CHARS:]

                # update conversation to include both guidance from RL and system outputs
                conversation.append({
                    "role": "user",
                    "content": "Command output:\n\n" + command_outputs,
                })

                           

