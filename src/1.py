from openai import OpenAI

client = OpenAI(
    api_key="sk-fexmhvxdiroabdqhndwdojnpeabvvzusjpcxzkmltwfhcytz",
    base_url="https://api.siliconflow.cn/v1",
)

resp = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.2",
    messages=[{"role": "user", "content": "hello"}],
)

print(resp.choices[0].message.content)