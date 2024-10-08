from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai", st.session_state.chosenModel)

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def create_llm(model):
    print(f"it is {model}!")
    return ChatOllama(
        model=model,
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(model=st.session_state.chosenModel)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role,model):
    st.session_state["messages"].append({"message": message, "role": role, "model":model})


def send_message(message, role, model, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if role == "ai":
            st.caption(model)
    if save:
        save_message(message, role, model)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            message["model"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
    
    Context: {context}
    Question:{question}
    """
)


st.title("PrivateGPT")

st.markdown(
    """

    ### 🤖 개인 GPT 봇

    해당 섹션은 개발자가 개인 챗봇 사용 용도입니다.
    
    사이드바에 다음 정보를 입력하세요:
    1. 📄 파일 업로드
    2. 💡 원하는 모델 선택

    개인만의 GPT를 써보시는 건 어떠세요?
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    chosenModel = st.selectbox("원하는 모델을 선택하세요.", ("mistral:latest", "llama3.1"),
                 placeholder="모델 선택 중...")
    
    # 모델이 변경될 때 llm 업데이트
    if "llm" not in st.session_state or st.session_state.chosenModel != chosenModel:
        st.session_state.llm = create_llm(chosenModel)
        st.session_state.chosenModel = chosenModel

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", st.session_state.chosenModel, save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human", st.session_state.chosenModel)
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | st.session_state.llm
        )
        print(message)
        with st.chat_message("ai"):
            chain.invoke(message)
            st.caption(st.session_state.chosenModel)


else:
    st.session_state["messages"] = []