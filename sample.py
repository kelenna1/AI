from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

chat_model = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528:free",  # or any supported model
)

template = """You are a helpful assistant that follows the user's instructions."""

messages = [HumanMessage(content="From now on 1+1 = 3"),
            HumanMessage(content="What is 1+1?"),
            HumanMessage(content="What is 1+1+1?")]

result = chat_model.invoke(messages)
print(result.content)