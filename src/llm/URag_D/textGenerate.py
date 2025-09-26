from pydantic_ai import Agent, RunContext
from src.llm.llm_kernel import KERNEL, GoogleConfig
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import json

load_dotenv()

# 1) Thử dùng cấu hình đã lưu (nếu UI từng lưu bằng KERNEL.save_active_config)
loaded = KERNEL.load_active_config()

# 2) Nếu chưa có active_config thì ép Google mặc định (bạn có thể bỏ khối này nếu UI luôn set)
if loaded is None:
    KERNEL.set_active_config(
        GoogleConfig(model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
    )

# 3) Lấy model đang active; cho phép override api_key/model_name qua ENV
model = KERNEL.get_active_model(
    model_name=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
)

@dataclass
class MyDeps:
    listChunks: list[str]

def build_prompt(ctx: RunContext[MyDeps])->str:
    data_json = json.dumps(ctx.deps.listChunks, ensure_ascii=False)
    return """You are a specialist in generating data, you can change a json file including
    chunks to an other json file with a meaningful sentence from each given chunk following these rules:
    **RULES** 
    1. The output must have json schema. (list of dictionaries)
    2. The value must be a dictionary with "original"(key) and "transformed"(value).
    3. Return strictly JSON, without any Markdown code fences.

    **Example**
    Given input:
        {
        "chunks": [
                "The solar system consists of the Sun",
                "and all the celestial bodies that orbit it,",
                "including planets, moons, asteroids, and comets.",
                "Scientists study these objects",
                "to understand the formation and evolution",
                "of our cosmic neighborhood."
            ]
        }
    Corrected output:
        [
            {
            "original": "The solar system consists of the Sun",
            "transformed": "The solar system is made up of the Sun as its central star."
            },
            {
            "original": "and all the celestial bodies that orbit it,",
            "transformed": "It also includes all the celestial bodies that revolve around the Sun."
            },
            {
            "original": "including planets, moons, asteroids, and comets.",
            "transformed": "These celestial bodies include planets, moons, asteroids, and comets."
            },
            {
            "original": "Scientists study these objects",
            "transformed": "Scientists carefully observe and study these objects"
            },
            {
            "original": "to understand the formation and evolution",
            "transformed": "to gain insights into how the solar system formed and evolved over time"
            },
            {
            "original": "of our cosmic neighborhood.",
            "transformed": "and to better understand the structure of our cosmic neighborhood."
            }
        ]
    ****

    You will be given a string with json format. Generating it following the above rules and making an output
    with string in json format.

    Your input is:
    """ + data_json

worker1 = Agent(
    model=model,
    deps_type=MyDeps
)
worker2 = Agent(
    model=model,
    deps_type=MyDeps
)

@worker1.system_prompt
def get_prompt1(ctx: RunContext[MyDeps]):
    return build_prompt(ctx)

@worker2.system_prompt
def get_prompt2(ctx: RunContext[MyDeps]):
    return build_prompt(ctx)

def get_data_from_json_file(file_name: str) -> dict:
    with open(file_name, "r", encoding="utf-8") as f:
        data = (json.load(f))
    return data

def parsed(json_output: str) -> None:
    OutputFile = "src/llm/URag_D/received_output.json"
    data = json.loads(json_output)
    if  os.path.exists(OutputFile):
        with open(OutputFile, "r", encoding="utf-8") as f:
            existed_data = json.load(f)

        if isinstance(existed_data, list):
            existed_data.extend(data)
        else:
            existed_data = [existed_data, data]
        
        with open(OutputFile, "w", encoding="utf-8") as f:
            json.dump(existed_data, f, ensure_ascii=False, indent=2)
    else:
        with open(OutputFile, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def checking_output(raw_output: str) -> str:
    for i in ["```json", "```", "'''json", "'''"]:
        raw_output = raw_output.replace(i, "")
    return raw_output.strip()