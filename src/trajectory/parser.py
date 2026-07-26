import re
import json
import chardet
import ast

from pathlib import Path

from .schema import (
    Trajectory,
    Message,
    ToolCall,
    ToolResult,
    Reward
)


# ===============================
# Encoding
# ===============================

def read_text_auto(path: Path):

    raw = path.read_bytes()

    result = chardet.detect(raw)

    encoding = result.get(
        "encoding",
        "utf-8"
    )

    confidence = result.get(
        "confidence",
        0
    )

    print(
        f"[Encoding] {path.name}: "
        f"{encoding}, confidence={confidence:.2f}"
    )


    return raw.decode(
        encoding,
        errors="replace"
    )



def fix_mojibake(text):

    """
    修复常见中文环境乱码

    鈥檇 -> ’
    鈥檓 -> ’
    鈥檛 -> ’
    """

    if not text:
        return text


    replacements = {

        "鈥檇": "’d",
        "鈥檓": "’m",
        "鈥檚": "’s",
        "鈥檛": "’t",
        "鈥檝": "’v",
        "鈥檒": "’l",

        "擨鈥檓": "I’m",
        "擨鈥檒": "I’ll",

        "鉁?": "✓",
        "鈫?": "→"
    }


    for k, v in replacements.items():

        text = text.replace(
            k,
            v
        )


    return text



# ===============================
# Step
# ===============================

def extract_step(line):

    m = re.search(
        r"Step (\d+)",
        line
    )

    if m:

        return int(
            m.group(1)
        )

    return -1



# ===============================
# Message
# ===============================

def extract_message_content(
        lines,
        start
):


    content = []


    for i in range(
        start,
        min(
            start + 20,
            len(lines)
        )
    ):


        line = lines[i]


        if "content:" in line:

            content.append(

                line.split(
                    "content:",
                    1
                )[1]
                .strip()

            )

            continue



        stop = [

            "is_final_chunk:",
            "ToolCalls:",
            "ToolCall",
            "2026-",
            "DEBUG"
        ]


        if any(
            x in line
            for x in stop
        ):

            break



        if content:

            content.append(
                line.strip()
            )



    result = " ".join(
        content
    )


    return fix_mojibake(
        result
    )



# ===============================
# Tool Call
# ===============================

def parse_tool_calls(
        text,
        step
):

    results = []


    pattern = re.compile(
        r"name:\s*([a-zA-Z0-9_]+)"
        r"\s*arguments:\s*(\{.*?\})"
    )


    matches = pattern.findall(
        text,
    )


    for name, args in matches:


        try:

            arguments = json.loads(
                args
            )

        except:

            arguments = {}


        results.append(

            ToolCall(

                step=step,

                tool_name=name,

                arguments=arguments
            )
        )


    return results



# ===============================
# Tool Result
# ===============================

def extract_tool_result(
        lines,
        start
):

    content=[]


    for i in range(
        start,
        min(
            start+80,
            len(lines)
        )
    ):


        line=lines[i]


        if "content:" in line:

            content.append(

                line.split(
                    "content:",
                    1
                )[1]
                .strip()

            )


        elif content:

            if (
                "requestor="
                in line
                or
                "2026-"
                in line
            ):

                break


            content.append(
                line.strip()
            )



    return fix_mojibake(
        " ".join(content)
    )



# ===============================
# Main parser
# ===============================

def parse_task_log(
        log_path: Path,
        task_id: str
):


    trajectory = Trajectory(
        task_id=str(task_id)
    )


    text = read_text_auto(
        log_path
    )


    lines = text.splitlines()


    current_step=-1



    for i,line in enumerate(lines):


        step = extract_step(
            line
        )


        if step != -1:

            current_step=step



        # -----------------------
        # Message
        # -----------------------

        if "From role:" in line:


            if "Role.AGENT" in line:

                role="assistant"

            elif "Role.USER" in line:

                role="user"

            else:

                continue



            content = extract_message_content(
                lines,
                i+1
            )


            trajectory.messages.append(

                Message(

                    step=current_step,

                    role=role,

                    content=content
                )
            )



        # -----------------------
        # Tool Call
        # -----------------------

        if "ToolCall (from assistant)" in line:


            block=" ".join(

                lines[
                    i:min(
                        i+8,
                        len(lines)
                    )
                ]

            )


            calls=parse_tool_calls(
                block,
                current_step
            )


            if calls:

                trajectory.tool_calls.extend(
                    calls
                )

            else:

                trajectory.tool_calls.append(

                    ToolCall(
                        step=current_step
                    )

                )



        # -----------------------
        # Tool Result
        # -----------------------

        if "Message: ToolMessage" in line:


            result=extract_tool_result(
                lines,
                i+1
            )


            trajectory.tool_results.append(

                ToolResult(

                    step=current_step,

                    content=result
                )

            )


    return trajectory



# ===============================
# Reward
# ===============================

def add_reward(
        trajectory,
        summary_path: Path
):


    if not summary_path.exists():

        return trajectory



    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as f:

        data=json.load(f)



    trajectory.reward=Reward(

        overall=data.get(
            "reward",
            0.0
        ),

        db_reward=data.get(
            "db_reward",
            0.0
        ),

        nl_assertion_reward=data.get(
            "nl_assertion_reward",
            0.0
        )
    )


    return trajectory



# ===============================
# Task loader
# ===============================

def parse_task_trajectory(
        task_dir: Path
):


    task_id=task_dir.name.replace(
        "task_",
        ""
    )


    artifact_root=(

        task_dir /
        "tau2_artifacts" /
        "artifacts" /
        f"task_{task_id}"

    )


    sims=list(

        artifact_root.glob(
            "sim_*"
        )

    )


    if not sims:

        raise RuntimeError(
            f"No sim found: {task_dir}"
        )


    sim_dir=sims[0]


    trajectory=parse_task_log(

        sim_dir /
        "task.log",

        task_id
    )


    trajectory=add_reward(

        trajectory,

        task_dir /
        "summary.json"

    )


    return trajectory