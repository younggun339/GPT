from langchain.storage import LocalFileStore
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

has_transcript = os.path.exists("./.cache/podcast.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )

@st.cache_data  # Caching the summary
def generate_summary(_docs):
    # First summarization step
    first_summary_prompt = ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:                
        """
    )
    first_summary_chain = first_summary_prompt | llm | StrOutputParser()

    summary = first_summary_chain.invoke(
        {"text": docs[0].page_content},
    )

    # Refine summary for subsequent documents
    refine_prompt = ChatPromptTemplate.from_template(
        """
        Your job is to produce a final summary.
        We have provided an existing summary up to a certain point: {existing_summary}
        We have the opportunity to refine the existing summary (only if needed) with some more context below.
        ------------
        {context}
        ------------
        Given the new context, refine the original summary.
        If the context isn't useful, RETURN the original summary.
        """
    )
    refine_chain = refine_prompt | llm | StrOutputParser()

    for i, doc in enumerate(docs[1:]):
        summary = refine_chain.invoke(
            {
                "existing_summary": summary,
                "context": doc.page_content,
            }
        )
    return summary


@st.cache_data
def get_memory():
    return ConversationSummaryBufferMemory(
        llm=ChatOpenAI(temperature=0.1),
        max_token_limit=120,
        return_messages=True,
    )

# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state["memory"] = get_memory()

memory = st.session_state["memory"]

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

st.markdown(
    """
## 💼 MeetingGPT

### 📹 비디오 분석 챗봇

영상의 대화 내용을 보지 않고 요약해보세요!

사이드바에 다음 정보를 입력하세요:
1. 📹 비디오

이러한 기능을 이용할 수 있습니다:
- 📝 대본 생성
- 📊 요약 제공
- 💬 내용에 관한 Q&A 챗봇

그럼 시작해볼까요? 🚀
"""
)

if "transcript_path" not in st.session_state:
    st.session_state["transcript_path"] = ""


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )



if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        st.session_state["transcript_path"]= transcript_path
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

  
    with transcript_tab:

        with open(transcript_path, "r") as file:
            st.write(file.read())
            

    with summary_tab:
        start = st.button("Generate summary")
        if start:

            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)
            
            with st.status("Summarizing...") as status:
                summary = generate_summary(docs)  # Cached function is called
                st.write(summary)
            st.write(summary)
            

    with qa_tab:
        # docs = retriever.invoke("do they talk about marcus aurelius?")
            qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Answer the question using ONLY the following context and conversation history. If you don't know the answer, just say you don't know. DON'T make anything up.
                    
                    Context: {context}
                    Conversation History: {history}
                    """,
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

            retriever = embed_file(transcript_path)

            send_message("준비됐어요! 물어보세요!", "ai", save=False)
            paint_history()

   # Now move the Q&A section outside of tabs



message = st.chat_input("영상에 관해 궁금한걸 물어보세요...")
if message:
    history = memory.load_memory_variables({})["history"]
    print(history)
    send_message(message, "human")
    qa_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: memory.load_memory_variables({})["history"]),
        }
        | qa_prompt
        | llm
    )
    with st.chat_message("ai"):
        result = qa_chain.invoke(message)
        memory.save_context(
            {"input": message},
            {"output": result.content},
        )
        st.session_state["memory"] = memory
        print(result.content)
        print(memory.load_memory_variables({})["history"])
else:
    st.session_state["messages"] = []
