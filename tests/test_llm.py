from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.llm import get_chat_model, get_embedding_model
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class Joke(BaseModel):
    '''A joke with a score'''

    joke: str = Field(description='the joke content')
    score: int = Field(description='score of the joke', le=10, ge=0)
strucuted_model = get_chat_model().with_structured_output(Joke)

res = strucuted_model.invoke(
    [
        SystemMessage("You're suppose to tell a joke with a func score between 0-10. Use compatible JSON only"),
        HumanMessage("What's an AI?")
    ]
)

print(res)

print(f"\n\n\n {type(res)}")



embedding_model=get_embedding_model()

queen=embedding_model.embed_query('I AM A QUEEN')

king=embedding_model.embed_query('I AM A KING')

boy=embedding_model.embed_query('I AM A BOY')

girl=embedding_model.embed_query('I AM A GIRL')


print(f"{queen} \n {king} \n {boy} \n {girl}")

