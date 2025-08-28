import os  
# os: lets Python talk to your operating system (like setting environment variables)
import getpass 
# getpass: lets you ask the user for input without showing it on screen (so passwords/keys stay hidden)

try :
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
os.environ["LANGSMITH_TRACING"] = "true"
# Creates an environment variable LANGSMITH_TRACING=true.
# This just tells LangSmith (LangChain’s tracing tool) to collect debug logs.
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key (optional): ")
# If LANGSMITH_API_KEY is missing, prompt the user to type it.

# That way, you’ll always have the key set either from .env or manual input.
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass('Enter your LangSmith Project (default="default"): ') or "default"

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
from langchain.chat_models import init_chat_model
# init_chat_model: quickly create a chat model (like Gemini, GPT, Claude).
from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate: helps you build prompts with variables safely.
from langchain_core.output_parsers import JsonOutputParser
# JsonOutputParser: tells the model to return JSON and parses it to a Python dict.
from pydantic import BaseModel, Field
# PydanticOutputParser: similar but returns a Pydantic object (instead of dict).
# BaseModel, Field: used to define structured schemas with validation.
from langchain_core.runnables import RunnableLambda
# RunnableLambda: lets you wrap custom Python functions into LangChain chains (not used in this script).
from langchain_core.output_parsers import PydanticOutputParser 

model=init_chat_model("gemini-2.5-flash", model_provider="google_genai")
# Creates a chat model object for Google Gemini 2.5 Flash.

# This model object can now be called with prompts.
class TranslationResult(BaseModel):
    # Creates a Pydantic model (a structured Python class) with strict fields:
    language:str=Field(...,description="Target language name")
    # language → target language (Italian, Spanish, etc).
    original_text:str=Field(...,description="Original input text")
    # original_text → the input text.
    translated_text:str=Field(...,description="translated_text")
    # translated_text → the translated result.
    confidence:str=Field(...,description="A value between 0 and 1 representing translation confidence")
    # confidence → how confident the AI is (0–1).
    notes: str = Field(..., description="Any short notes or caveats about the translation")
    # notes → any extra comments.
# Field(..., description="...") = ... means required field, and description helps generate better JSON instructions for the model.


# parser=PydanticOutputParser(pydantic_object=TranslationResult)
parser=JsonOutputParser(pydantic_object=TranslationResult)
# You’re using JsonOutputParser. This will:

# Tell the model to output JSON in that schema.
# Parse it into a Python dict.
# Later, you manually convert that dict into a TranslationResult.
system_template="""
you are a helpful translator.
Return ONLY valid JSON that matches the given schema.
{format_instructions}
"""
user_template="""Translate the text below into {language}.
Also provide a confidence score between 0 and 1 and any brief notes.
Text:{text}

"""
prompt=ChatPromptTemplate.from_messages(
    # from_messages: builds a structured chat with both system + user roles.
    [
        ("system",system_template),
        ("user",user_template)
    ]
).partial(format_instructions=parser.get_format_instructions())
# Creates a structured chat prompt (system + user roles).

# partial(...) fills in format_instructions once so you don’t need to pass it each time.

# BUILD THE CHAIN
chain=prompt | model | parser
# prompt takes your inputs and builds the formatted messages.

# model sends them to Gemini.

# parser converts the response into a TranslationResult Python object.

inputs={
    "language":"italian",

    "text":"hi! how are you doing today?"
}
result_dict=chain.invoke(inputs)
# Runs the chain end-to-end.
# Gemini replies with JSON.
# Parser converts it into a Python dict.
# {language} = "italian"

# {text} = "Hi! How are you doing today?"

# Sends it through the model and parser.
# Example output:
# {
#   "language": "Italian",
#   "original_text": "hi! how are you doing today?",
#   "translated_text": "Ciao! Come stai oggi?",
#   "confidence": "0.99",
#   "notes": "Informal register used."
# }

result=TranslationResult.model_validate(result_dict)
# Takes the dict and converts it into a TranslationResult object.
# Now you can access fields like .translated_text or .confidence safely.
# ydantic will also check the data types (e.g., if confidence isn’t a number, it would warn).
# Fills the template:

print("parsed object:",result)
print("translated text:",result.translated_text)
print("confidence",result.confidence)

# If you need a plain dict (e.g., to JSON-serialize):
as_dict=result.model_dump()
# model_dump() = convert Pydantic object → plain Python dict.
print("as dict",as_dict)

# If you need a JSON string:
as_json=result.model_dump_json(indent=2)  # pretty-printed JSON
print("as json",as_json)
# as josn
