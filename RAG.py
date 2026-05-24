import getpass
import os
import dotenv
from langchain_groq import ChatGroq

dotenv.load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Corrected initialization
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
question = input("Enter your question: ")
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to JAP. Translate the user sentence. "
        "explain it word by word.",
    ),
    ("human", question),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
