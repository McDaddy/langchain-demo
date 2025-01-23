import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv

load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv('DASHSCOPE_API_KEY')

# 聊天机器人案例
# 创建模型
model = Tongyi()

# 准备测试数据 ，假设我们提供的文档数据如下：
documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鲨鱼是非常凶残的东西，有极强的攻击性。",
        metadata={"source": "鱼类文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

# 实例化一个向量数空间
vector_store = Chroma.from_documents(documents, embedding=DashScopeEmbeddings())

# 相似度的查询: 返回相似的分数， 分数越低相似度越高
# print(vector_store.similarity_search_with_score('咖啡猫'))

# 检索器: bind(k=1) 返回相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search).bind(k=3)

# print(retriever.batch(['咖啡猫', '鲨鱼']))


# 提示模板
message = """
使用提供的上下文仅回答这个问题:
{question}
上下文:
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

# RunnablePassthrough允许我们将用户的问题之后再传递给prompt和model。
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model

resp = chain.invoke('请介绍一下动物？')

print(resp)

