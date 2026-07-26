from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Message:
    step: int
    role: str
    content: str


@dataclass
class ToolCall:
    step: int
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    step: int
    content: str
    tool_name: str = ""


@dataclass
class Reward:
    overall: float = 0.0
    db_reward: float = 0.0
    nl_assertion_reward: float = 0.0


@dataclass
class Trajectory:

    task_id: str

    messages: List[Message] = field(default_factory=list)

    tool_calls: List[ToolCall] = field(default_factory=list)

    tool_results: List[ToolResult] = field(default_factory=list)

    reward: Reward = field(default_factory=Reward)


    def to_dict(self):

        return {

            "task_id": self.task_id,

            "messages": [
                {
                    "step": m.step,
                    "role": m.role,
                    "content": m.content
                }
                for m in self.messages
            ],


            "tool_calls": [
                {
                    "step": t.step,
                    "tool_name": t.tool_name,
                    "arguments": t.arguments
                }
                for t in self.tool_calls
            ],


            "tool_results": [
                {
                    "step": t.step,
                    "tool_name": t.tool_name,
                    "content": t.content
                }
                for t in self.tool_results
            ],


            "reward": {

                "overall": self.reward.overall,

                "db_reward":
                    self.reward.db_reward,

                "nl_assertion_reward":
                    self.reward.nl_assertion_reward
            }
        }