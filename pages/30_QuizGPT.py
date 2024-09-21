import json

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.

    You format exam questions into JSON format. Answers with (o) are the correct ones.

    In your JSON format, for each incorrect answer, please provide an appropriate comment explaining why the answer is incorrect.
        
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
 {{ 
    "questions": [
        {{
            "question": "What is the color of the ocean?",
            "answers": [
                {{
                    "answer": "Red",
                    "correct": false,
                    "comment": "Red is not the typical color associated with large bodies of water."
                }},
                {{
                    "answer": "Yellow",
                    "correct": false,
                    "comment": "Yellow is uncommon for oceans, usually representing shallow waters or algae blooms."
                }},
                {{
                    "answer": "Green",
                    "correct": false,
                    "comment": "Green can sometimes appear in oceans due to algae, but it's not the primary color."
                }},
                {{
                    "answer": "Blue",
                    "correct": true,
                    "comment": "This is because the color of the sky is reflected in the sea."
                }}
            ]
        }},
        {{
            "question": "What is the capital of Georgia?",
            "answers": [
                {{
                    "answer": "Baku",
                    "correct": false,
                    "comment": "Baku is the capital of Azerbaijan, not Georgia."
                }},
                {{
                    "answer": "Tbilisi",
                    "correct": true,
                    "comment": "Tbilisi is the correct capital of Georgia."
                }},
                {{
                    "answer": "Manila",
                    "correct": false,
                    "comment": "Manila is the capital of the Philippines, not Georgia."
                }},
                {{
                    "answer": "Beirut",
                    "correct": false,
                    "comment": "Beirut is the capital of Lebanon, not Georgia."
                }}
            ]
        }},
        {{
            "question": "When was Avatar released?",
            "answers": [
                {{
                    "answer": "2007",
                    "correct": false,
                    "comment": "Avatar was not released in 2007, but in a later year."
                }},
                {{
                    "answer": "2001",
                    "correct": false,
                    "comment": "2001 is too early for Avatar's release."
                }},
                {{
                    "answer": "2009",
                    "correct": true,
                    "comment": "Avatar was released in December 2009."
                }},
                {{
                    "answer": "1998",
                    "correct": false,
                    "comment": "1998 is far too early for Avatar, as it was still in development."
                }}
            ]
        }},
        {{
            "question": "Who was Julius Caesar?",
            "answers": [
                {{
                    "answer": "A Roman Emperor",
                    "correct": true,
                    "comment": "Julius Caesar was a Roman military leader and statesman who played a critical role in the fall of the Roman Republic."
                }},
                {{
                    "answer": "Painter",
                    "correct": false,
                    "comment": "Julius Caesar was not known for painting."
                }},
                {{
                    "answer": "Actor",
                    "correct": false,
                    "comment": "Julius Caesar was not an actor."
                }},
                {{
                    "answer": "Model",
                    "correct": false,
                    "comment": "Caesar was a political leader, not a model."
                }}
            ]
        }}
    ]
}}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs



with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
    """
    ## 🎓 퀴즈 챗봇


    다음 자료로 퀴즈를 만들어 드립니다:
    - 🌐 위키피디아 문서
    - 📁 업로드하신 파일

    사이드바에 다음 정보를 입력하세요:
    1. 파일 업로드하기 📤 또는
    2. 위키피디아에서 검색하기 🔍
    
    함께 학습 여정을 시작해볼까요? 🚀
    """
)
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for i, question in enumerate(response["questions"]):
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"radio_{i}" 
            )
            # Find the selected answer object and the correct answer object
            selected_answer = next((answer for answer in question["answers"] if answer["answer"] == value), None)
            correct_answer = next((answer for answer in question["answers"] if answer["correct"]), None)
            if selected_answer is not None:
                if selected_answer["correct"]:
                    # Show success message with the comment for the correct answer
                    st.success(f"Correct! {selected_answer['comment']}")
                else:
                    # Show error message with the comment for the incorrect answer and show the correct answer
                    st.error(f"Wrong! correct answer is {correct_answer['answer']}. {selected_answer['comment']}")
            
            # if {"answer": value, "correct": True} in question["answers"]:
            #     st.success("Correct!")
            # elif value is not None:
            #     correct_answer = next(answer["answer"] for answer in question["answers"] if answer["correct"])
            #     st.error(f"Wrong! 정답은 {correct_answer}입니다.")
        button = st.form_submit_button()