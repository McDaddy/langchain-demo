import os
from openai import OpenAI

client = OpenAI(
    api_key = os.environ.get("ARK_API_KEY"),
    base_url = "https://ark-cn-beijing.bytedance.net/api/v3",
)

# print("----- standard request -----")
# completion = client.chat.completions.create(
#     model = "ep-20250205111111-f5mwd",  # your model endpoint ID
#     messages = [
#         {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
# )
# print(completion.choices[0].message.content)

print("----- streaming request -----")
stream = client.chat.completions.create(
    model = "ep-20250205110448-bqkc4",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
    stream=True
)

for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()