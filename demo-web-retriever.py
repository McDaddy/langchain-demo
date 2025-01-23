import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv

load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv('DASHSCOPE_API_KEY')

# 聊天机器人案例
# 创建模型
model = Tongyi()

# 1、加载数据: 一篇博客内容数据
loader = WebBaseLoader(
    web_paths=['https://www.kuimo.top/blog/2023/front-end'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('prose max-w-none pb-8 pt-10 dark:prose-invert'))
    )
)

docs = loader.load()

# print(len(docs))
# print(docs)

# 2、大文本的切割
# text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = splitter.split_documents(docs)

# 2、存储
vectorstore = Chroma.from_documents(documents=splits, embedding=DashScopeEmbeddings())

# 3、检索器
retriever = vectorstore.as_retriever()

# 整合

# 创建一个问题的模板
system_prompt = """你是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话，并保持答案简洁.\n
{context}
"""
prompt = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),  #
        ("human", "{input}"),
    ]
)

# 得到chain
chain1 = create_stuff_documents_chain(model, prompt)

# chain2 = create_retrieval_chain(retriever, chain1)

# resp = chain2.invoke({'input': "What is Task Decomposition?"})
#
# print(resp['answer'])

'''
注意：
一般情况下，我们构建的链（chain）直接使用输入问答记录来关联上下文。但在此案例中，查询检索器也需要对话上下文才能被理解。

解决办法：
添加一个子链(chain)，它采用最新用户问题和聊天历史，并在它引用历史信息中的任何信息时重新表述问题。这可以被简单地认为是构建一个新的“历史感知”检索器。
这个子链的目的：让检索过程融入了对话的上下文。
'''

# 创建一个子链
# 子链的提示模板
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)

# 创建一个子链
history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

# 保持问答的历史记录
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建父链chain: 把前两个链整合
chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# 第一轮对话
resp1 = result_chain.invoke(
    {'input': '当代市占率最高的浏览器是什么?'},
    config={'configurable': {'session_id': 'zs123456'}}
)

print(resp1['answer'])

# 第二轮对话
resp2 = result_chain.invoke(
    {'input': '它是什么时候诞生的?'},
    config={'configurable': {'session_id': 'zs123456'}}
)

print(resp2['answer'])