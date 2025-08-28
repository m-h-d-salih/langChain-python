import getpass
import os


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"]=getpass.getpass(
        prompt="Eneter your LangSmit API key (optional):"
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"]=getpass.getpass(
        prompt='Enter your LangSmith project Name (default="default"):'
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"]="default"

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
from langchain.chat_models import init_chat_model
model=init_chat_model("gemini-2.5-flash",model_provider="google_genai")
from langchain_core.messages import HumanMessage,SystemMessage

messages=[
    SystemMessage("Translate the following from English to Italian"),
    HumanMessage("Hi !")
]
res=model.invoke(messages)
res2=model.invoke("hello")
res3=model.invoke([{"role":"user","content":"Hello"}])
res4=model.invoke([HumanMessage(content="hello")])
print(res4)