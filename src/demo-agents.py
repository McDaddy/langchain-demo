import os
from langchain.agents import initialize_agent, Tool
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv

load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv('DASHSCOPE_API_KEY')

# 聊天机器人案例
# 创建模型
model = Tongyi()

def search_order(input: str) -> str:
    #  实现一个搜索订单状态的 Tool
    return "订单状态：已发货；发货日期：2023-01-01；预计送达时间：2023-01-10"

def recommend_product(input: str) -> str:
    #  实现一个推荐产品的 Tool
    return "红色连衣裙"

def faq(intput: str) -> str:
    #  实现一个 FAQ 的 Tool
    return "7天无理由退货"

#  注册 Tool，并给出每个 Tool 的 Description
tools = [
    Tool(
        name = "Search Order",func=search_order, 
        description="useful for when you need to answer questions about customers orders"
    ),
    Tool(name="Recommend Product", func=recommend_product, 
         description="useful for when you need to answer questions about product recommendations"
    ),
    Tool(name="FAQ", func=faq,
         description="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."
    )
]

agent = initialize_agent(tools, model, agent="zero-shot-react-description", verbose=True)

# question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
# result = agent.run(question)
# print(result)

question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
result = agent.run(question)
print(result)