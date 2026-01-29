import spacy
from typing import Dict

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


MISTRAL_API_KEY = "41XefarZewGcSLzbHEnpsO9lrVeFDf5V"

nlp = spacy.load("en_core_web_sm")


def extract_entity(text: str):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}:
            return {"text": ent.text, "label": ent.label_}
    return None


def contains_pronoun(text: str):
    pronouns = {"he", "she", "his", "her", "they", "their", "it", "its"}
    return any(p in text.lower().split() for p in pronouns)


def is_entity_seeking(text: str):
    if not text.endswith("?"):
        return False
    if extract_entity(text):
        return False

    doc = nlp(text)
    return any(
        t.text in {"who", "what", "which"} or t.tag_ in {"JJS", "RBS"}
        for t in doc
    )


def promote_entity_from_text(text: str):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}:
            return {"text": ent.text, "label": ent.label_}
    return None



class SessionState:
    def __init__(self):
        self.history = InMemoryChatMessageHistory()
        self.entity = None


store: Dict[str, SessionState] = {}


def get_session_state(session_id: str) -> SessionState:
    if session_id not in store:
        store[session_id] = SessionState()
    return store[session_id]


llm = ChatMistralAI(
    model="mistral-small",
    temperature=0,
    mistral_api_key=MISTRAL_API_KEY
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful factual conversational assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm


conversation = RunnableWithMessageHistory(
    chain,
    lambda sid: get_session_state(sid).history,
    input_messages_key="input",
    history_messages_key="history"
)

def ask_bot(user_query: str, session_id: str = "default") -> dict:
    state = get_session_state(session_id)

    #  Entity detection
    detected = extract_entity(user_query)
    if detected:
        state.entity = detected

    #  Pronoun resolution
    if contains_pronoun(user_query) and state.entity:
        wiki_query = state.entity["text"]
        resolved_query = f"{user_query} (This refers to {wiki_query})"
    else:
        wiki_query = user_query
        resolved_query = user_query

    #  Wikipedia grounding
    wiki_info = ""
    try:
        wiki_info = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=1)
        ).run(wiki_query)
    except Exception:
        pass

    # Implicit entity promotion
    if state.entity is None and is_entity_seeking(user_query):
        promoted = promote_entity_from_text(wiki_info)
        if promoted:
            state.entity = promoted

    #   LLM call
    final_input = resolved_query
    if wiki_info:
        final_input += f"\n\nReference:\n{wiki_info}"

    result = conversation.invoke(
        {"input": final_input},
        config={"configurable": {"session_id": session_id}}
    )

    return {
        "answer": result.content,
        "entity": state.entity,
        "source": "Wikipedia" if wiki_info else "LLM"
    }
