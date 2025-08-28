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
    confidence:float=Field(...,description="A value between 0 and 1 representing translation confidence")
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

texts = [
    "Hi! How are you?",
    "Good morning!",
    "Where is the train station?",
]
jobs = [{"language": "italian", "text": t} for t in texts]
chain=prompt | model | parser
batch_results = chain.batch(jobs)
typed_results = [TranslationResult.model_validate(r) for r in batch_results]
for r in typed_results:
    print(r.translated_text, r.confidence)


inputs={
    "language":"italian",

    "text":"hi! how are you doing today?"
}
result_dict=chain.invoke(inputs)


result=TranslationResult.model_validate(result_dict)
msgs = prompt.invoke({"language": "italian", "text": "Tell me a short joke."}).to_messages()

print("Streaming:")
for chunk in model.stream(msgs):
    # each chunk is a partial message; print incrementally
    print(chunk.content, end="", flush=True)
print("\n--- done ---")

class DetectResult(BaseModel):
    source_language:str=Field(...,description="Detect langauge name")
detect_parser=JsonOutputParser(pydantic_object=DetectResult)
detect_system="""
Return ONLY JSON with the detected source language.
{format_instructions}
"""
detect_user="Detect the language of this text: {text}"

detect_prompt=ChatPromptTemplate.from_messages(
    [("system",detect_system),("user",detect_user)]
).partial(format_instructions=detect_parser.get_format_instructions())

LANG_ALIASES = {
    "italian": {"italian", "italiano", "it"},
    "english": {"english", "inglès", "en"},
    # add more as needed
}
def same_language(target: str, detected: str) -> bool:
    t = target.strip().lower()
    d = detected.strip().lower()
    aliases = LANG_ALIASES.get(t, {t})
    return d == t or d in aliases or d.startswith(t)
detect_chain=detect_prompt | model | detect_parser
def translate_with_detection(language:str,text:str)->TranslationResult:
    det:DetectResult=DetectResult.model_validate(detect_chain.invoke({'text':text}))
    if same_language(language, det.source_language):
        return TranslationResult(
            language=language,
            original_text=text,
            translated_text=text,
            confidence=1.0,
            notes="Already in target language; no translation performed.",
        )
        # otherwise do your translation chain
    data = chain.invoke({"language": language, "text": text})
    return TranslationResult.model_validate(data)
result = translate_with_detection("italian", "Ciao, come stai?")
print(result)

import time

def invoke_with_retry(callable_fn,args:dict,tries=3,delay=1.5):
    for i in range(tries):
        try:
            return callable_fn(args)
        except Exception as e:
            if i==tries-1:
                raise
            time.sleep(delay)

data=invoke_with_retry(chain.invoke,{"language": "italian", "text": "hello!"})
print(TranslationResult.model_validate(data))
