#!/usr/bin/env python
import os
import yaml
from pathlib import Path

import openai

def load_first_default_key(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "default" not in cfg or not isinstance(cfg["default"], list) or len(cfg["default"]) == 0:
        raise ValueError("config 中没有有效的 `default` 配置")

    # 这里我们只取第一个 default 条目
    first = cfg["default"][0]
    api_key = first.get("api_key")
    base_url = first.get("base_url") or first.get("api_base")

    if api_key is None:
        raise ValueError("default[0] 中没有找到 api_key 字段")

    return api_key, base_url

def main():
    config_path = os.environ.get("OPENAI_CLIENT_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("请先设置环境变量 OPENAI_CLIENT_CONFIG_PATH 指向 openai_configs.yaml")

    config_path = str(Path(config_path).expanduser().absolute())
    print(f"Using config file: {config_path}")

    api_key, base_url = load_first_default_key(config_path)
    print(f"Loaded api_key (prefix only): {api_key[:10]}******")
    if base_url:
        print(f"Loaded base_url: {base_url}")
    else:
        print("No base_url set, using OpenAI default")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)

    # 真正的测试：打一条很短的请求
    try:
        resp = client.chat.completions.create(
            model="gpt-4-1106-preview",   # 只要这个模型你的 key 有权限就可以
            messages=[
                {"role": "user", "content": "Just respond with: OK"}
            ],
            max_tokens=2,
        )
        print("API call success! Response:")
        print(resp.choices[0].message.content)
    except Exception as e:
        print("API call failed!")
        print(repr(e))

if __name__ == "__main__":
    main()
