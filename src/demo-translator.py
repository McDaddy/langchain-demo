import os

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.tongyi import Tongyi
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv('DASHSCOPE_API_KEY')

# 调用大语言模型
# 创建模型
model = Tongyi()

# 简单的解析响应数据
# 3、创建返回的数据解析器
parser = StrOutputParser()

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译成{language}'),
    ('user', "{text}")
])

# 4、得到链
chain = prompt_template | model | parser

# 把我们的程序部署成服务
# 创建fastAPI的应用
app = FastAPI(title='我的Langchain服务', version='V1.0', description='使用Langchain翻译任何语句的服务器')

add_routes(
    app,
    chain,
    path="/chainDemo",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)