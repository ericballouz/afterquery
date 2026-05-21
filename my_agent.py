import anthropic
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pathlib import Path
import json
import re

class MyAgent(BaseAgent):
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

    async def setup(self, environment: BaseEnvironment) -> None:
        # any setup commands to run in the container before the task
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        client = anthropic.Anthropic()

        # Your custom system prompt here
        prompt_path = Path("/Users/ericballouz/Desktop/AfterQuery/terminal-bench/agent_op/input_prompt_largest_eigenval.txt")
        system_prompt = prompt_path.read_text().strip() 

        trajectory = []
        messages = [{"role": "user", "content": instruction}]

        try:
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                )

                # Extract the command the model wants to run
                raw_output = response.content[0].text
                commands = self.extract_commands(raw_output)

                # Execute it in the container
                if len(commands) == 0:
                    # model finished, no command to run
                    trajectory.append({ 
                        "step": len(trajectory) + 1,
                        "assistant_reasoning": raw_output,
                        "command": None,
                        "stdout": None,
                        "stderr": None,
                    })
                    break
    
                for command in commands:
                    # update trajectory
                    result = await environment.exec(command)
                    trajectory.append({
                    "step": len(trajectory) + 1,
                    "command": command,
                    "assistant_reasoning": raw_output,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    })
                #print(f"Turn {len(trajectory)+1} stop_reason: {response.stop_reason}")
                #print(f"Commands found: {commands}")

                # Add to message history
                messages.append({"role": "assistant", "content": raw_output})
                # append all outputs together
                all_outputs = "\n".join(
                    f"$ {t['command']}\n{t['stdout'] or t['stderr'] or ''}"
                    for t in trajectory[-len(commands):]
                )
                messages.append({"role": "user", "content": all_outputs})

        except Exception as e:
                trajectory.append({"error": str(e)})
        
        finally:
            log_file = self.logs_dir / "trajectory.json"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(json.dumps(trajectory, indent=2))

