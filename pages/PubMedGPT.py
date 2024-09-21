from langchain.document_loaders import DataFrameLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st


import pandas as pd
import umap.umap_ as umap
from tqdm import tqdm
tqdm.pandas()
from Bio import Entrez
from sklearn.cluster import KMeans
import datetime


llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                # "source": doc.metadata["source"],
                # "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


# @st.cache_data(show_spinner="Loading website...")
# def load_website(url):
#     splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000,
#         chunk_overlap=200,
#     )
#     loader = SitemapLoader(
#         url,
#     )
#     loader.requests_per_second = 2
#     docs = loader.load_and_split(text_splitter=splitter)
#     vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
#     return vector_store.as_retriever()

@st.cache_data(show_spinner="Creating vector store...")
def create_vector_store(df):
    # OpenAI 임베딩 함수 정의
    embeddings = OpenAIEmbeddings()
    
    def get_embedding(text):
        return embeddings.embed_query(text)

    # DataFrame의 'Summary' 열에 임베딩 적용
    if 'Embeddings' not in df.columns:
        with st.spinner("Generating embeddings..."):
            df['Embeddings'] = df['Summary'].progress_apply(get_embedding)
        st.success("Embeddings generated successfully!")

    # 텍스트 분할기 설정
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=150,
    )

    # DataFrame을 문서로 변환
    loader = DataFrameLoader(df, page_content_column="Summary")
    documents = loader.load()

    # 문서 분할
    docs = splitter.split_documents(documents)

    # FAISS 벡터 저장소 생성
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, doc.metadata['Embeddings']) for doc in docs],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in docs]
    )

    return vector_store.as_retriever()


@st.cache_data(show_spinner="Searching Document....")
def search_pubmed(keyword, retmax=10):
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()

    # Fetch the records for the retrieved IDs
    id_list = record["IdList"]
    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = fetch_handle.read()
    fetch_handle.close()

    # Parse the records and extract the desired information
    pubmed_records = records.split("\n\n")
    data = []
    for record in pubmed_records:
        record_dict = {}
        lines = record.split("\n")
        part_of_abstract = False
        part_of_title = False
        for line in lines:
            if part_of_abstract == True:
                if line[0] == " ":
                    record_dict["Abstract"] += " " + line[6:].strip()
                else:
                    part_of_abstract = False
            elif part_of_title == True:
                if line[0] == " ":
                    record_dict["Title"] += " " + line[6:].strip()
                else:
                    part_of_title = False
            elif line.startswith("PMID"):
                record_dict["PMID"] = line[6:].strip()
            elif line.startswith("TI"):
                record_dict["Title"] = line[6:].strip()
                part_of_title = True
            elif line.startswith("AB"):
                record_dict["Abstract"] = line[6:].strip()
                part_of_abstract = True
            elif line.startswith("DP"):
                record_dict["Year"] = line[6:10]
            elif "Author" not in record_dict and line.startswith("FAU"):
                fau = line[6:].strip().split(', ')
                if len(fau) > 1:
                    record_dict["Author"] = f"{fau[1]} {fau[0]}"
                else:
                    record_dict["Author"] = line[6:].strip()
            elif line.startswith("TA"):
                record_dict["Journal"] = line[6:]
        if all(k in record_dict for k in ("Abstract", "PMID", "Title", "Year", "Author", "Journal")):
            data.append(record_dict)

    return data

# keyword = "Medical Education ChatGPT"
# search_results = search_pubmed(keyword, 30)
# df = pd.DataFrame(search_results)
# df

summary_prompt = ChatPromptTemplate.from_messages([("system", """
You are a research assistant, and will be provided with an abstract of a scientific paper. Write a concise 2-line summary of the main findings.
"""),("user", """{context}
""")])

@st.cache_data(show_spinner="Genarating Summary...")
def get_summary(keyword, numbers):
    search_results = search_pubmed(keyword, numbers)
    df = pd.DataFrame(search_results)
    summary_chain = summary_prompt | llm
   # 각 Abstract를 요약하는 함수
    def summarize_abstract(abstract):
        return summary_chain.invoke({"abstract": abstract})
    
    # tqdm을 사용한 progress_apply로 각 Abstract 요약
    df['Summary'] = df['Abstract'].progress_apply(summarize_abstract)
    
    return df


# df['Summary'] = df['Abstract'].progress_apply(get_summary)
# df


topic_prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a research assistant, and will be provided with a list of documents.\nBased on the information extract a single short but highly desriptive topic label. Answer with 1 topic label and make it less than 8 words. Make sure it is in the following format:\ntopic: <topic label>"""),
    ("user", """{context}""")
])

@st.cache_data(show_spinner="Getting Topic...")
def get_topic(_df):
    topic_chain = {"context" : create_vector_store }| topic_prompt | llm
    return topic_chain.invoke(_df)

# topics = []
# for i in range(k):
#   abstracts = ""
#   for _, row in df[df['Group']==i].iterrows():
#     abstracts += f"- {row['Summary'].strip()}\n"
#   topics.append(get_topic(abstracts))

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))