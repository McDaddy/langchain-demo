import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv

load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv('DASHSCOPE_API_KEY')

# 聊天机器人案例
# 创建模型
model = Tongyi()

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手。用{language}尽你所能回答所有问题。'),
    MessagesPlaceholder(variable_name='my_msg')
])

# 得到链
chain = prompt_template | model

# 保存聊天的历史记录
store = {}  # 所有用户的聊天记录都保存到store。key: sessionId,value: 历史聊天记录对象


# 此函数预期将接收一个session_id并返回一个消息历史记录对象。
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'  # 每次聊天时候发送msg的key
)

config = {'configurable': {'session_id': 'zs1234'}}  # 给当前会话定义一个sessionId

# 第一轮
resp1 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='你好啊！ 我是McDaddy')],
        'language': '中文'
    },
    config=config
)

print(resp1)

# 第二轮
resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='请问：我的名字是什么？')],
        'language': '中文'
    },
    config=config
)

print(resp2)

# 第3轮： 返回的数据是流式的
config = {'configurable': {'session_id': 'lis2323'}}  # 给当前会话定义一个sessionId
# 第3轮
resp3 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='请问：我的名字是什么？')],
        'language': '中文'
    },
    config=config
)

print(resp3)


for resp in do_message.stream({'my_msg': [HumanMessage(content='请给我讲一个笑话？')], 'language': '中文'},
                              config=config):
    # 每一次resp都是一个token
    print(resp)
