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
    return choose_chain.invoke(
        {
            "question": question,
            "answers": answers,
        }
    )


@st.cache_data(show_spinner="Creating vector store...")
def create_vector_store_QA(df):
    # OpenAI 임베딩 함수 정의
    embeddings = OpenAIEmbeddings()
    
    def get_embedding(text):
        return embeddings.embed_query(text)

    # DataFrame의 'Summary' 열에 임베딩 적용
    if 'Embeddings_Ab' not in df.columns:
        with st.spinner("Generating embeddings..."):
            df['Embeddings_Ab'] = df['Abstract'].progress_apply(get_embedding)
        st.success("Embeddings generated successfully!")

    # 텍스트 분할기 설정
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=150,
    )

    # DataFrame을 문서로 변환
    loader = DataFrameLoader(df, page_content_column="Abstract")
    documents = loader.load()

    # 문서 분할
    docs = splitter.split_documents(documents)

    # FAISS 벡터 저장소 생성
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, doc.metadata['Embeddings_Ab']) for doc in docs],
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
        return summary_chain.invoke({"context": abstract})
    
    # tqdm을 사용한 progress_apply로 각 Abstract 요약
    print(df['Abstract'])
    df['Summary'] = df['Abstract'].progress_apply(summarize_abstract)
    print(df['Summary'])
    
    return df


# df['Summary'] = df['Abstract'].progress_apply(get_summary)
# df


@st.cache_data(show_spinner="Embedding summary...")
def embedding_summary(df):
    # OpenAI 임베딩 함수 정의
    embeddings = OpenAIEmbeddings()
    
    def get_embedding_topic(text):
        return embeddings.embed_query(text.content)

    # DataFrame의 'Summary' 열에 임베딩 적용
    if 'Embeddings_Sum' not in df.columns:
        with st.spinner("Generating embeddings..."):
            df['Embeddings_Sum'] = df['Summary'].progress_apply(get_embedding_topic)
        st.success("Embeddings generated successfully!")


    return df

@st.cache_data(show_spinner="Reduce demension...")
def reduce_demension(df):
    reducer = umap.UMAP(n_neighbors=3, n_components=3)
    u = reducer.fit_transform(df['Embeddings_Sum'].tolist())
    return u

@st.cache_data(show_spinner="Clustering...")
def cluster(k, u, df):
    k = int(k) 
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(u)
    labels = kmeans.labels_
    df['Group'] = labels
    print(f"cluster df : {df}")
    return df

topic_prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a research assistant, and will be provided with a list of documents.\nBased on the information extract a single short but highly desriptive topic label. Answer with 1 topic label and make it less than 8 words. Make sure it is in the following format:\ntopic: <topic label>"""),
    ("user", """{context}""")
])

@st.cache_data(show_spinner="Getting Topic...")
def get_topic(_df, k):
    k = int(k)
    data = embedding_summary(_df)
    u = reduce_demension(data)
    last_df = cluster(k, u, data)
    topic_chain = topic_prompt | llm
    print(f"df : {last_df}")
    
    topics = []
    for i in range(k):
        abstracts = ""
        for _, row in last_df[last_df['Group']==i].iterrows():
            abstracts += f"- {row['Summary'].content.strip()}\n"
        topic_comment = topic_chain.invoke({"context" : abstracts})
        topic = [topic.replace("topic: ", "").strip() for topic in topic_comment.content.split('\n') if topic.strip()]
        topics.append(topic)
    
    # "topic: " 접두사 제거 및 리스트로 변환
    return last_df, topics

def output_parser(df, topics):
    output = []
    for i, topic in enumerate(topics):
        group_df = df[df['Group'] == i]
        group_data = {
            'Topic': topic,
            'Group': i,
            'Papers': []
        }
        for _, row in group_df.iterrows():
            paper = {
                'Author': row['Author'],
                'Year': row['Year'],
                'Summary': row['Summary'].content.strip()
            }
            group_data['Papers'].append(paper)
        output.append(group_data)
    return output

# topics = []
# for i in range(k):
#   abstracts = ""
#   for _, row in df[df['Group']==i].iterrows():
#     abstracts += f"- {row['Summary'].strip()}\n"
#   topics.append(get_topic(abstracts))

st.set_page_config(
    page_title="PubMedGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # PubMedGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    keyword = st.text_input(
        "Write down a Keyword",
        placeholder="ex) DNA, MS...",
    )
    number = st.text_input(
        "Write down a number",
        placeholder="ex) 5, 10, 15..."
    )
    k = st.text_input("Write down a clustering number",
                            placeholder="ex) 1, 2...")


if keyword:
    # df = get_summary(keyword, number)
    summary_tab, qa_tab = st.tabs(
        [
            "Summary",
            "Q&A",
        ]
    )
    with summary_tab:
        start = st.button("Genrate summary")
        if start:
            df = get_summary(keyword, number)
            last_df, topics = get_topic(df, k)
            print(last_df)
            print(topics)
            parsed_output = output_parser(last_df, topics)
            print(parsed_output)
             # Streamlit을 사용하여 결과 표시
            for group in parsed_output:
                st.subheader(f"Group {group['Group']}: {group['Topic']}")
                for paper in group['Papers']:
                    st.write(f"- ({paper['Author']}, {paper['Year']}) {paper['Summary']}")
                st.write("---")

    with qa_tab:
        st.button("아무거나.")
    # search_results = search_pubmed(keyword, number)
    # df = pd.DataFrame(search_results)
    # retriever = create_vector_store(df)
query = st.text_input("Ask a question to the website.")
if query: 
    search_results = search_pubmed(keyword, number)
    df = pd.DataFrame(search_results)
    retriever = create_vector_store_QA(df)
    chain = ({
        "docs": retriever,
        "question" : RunnablePassthrough(),
    }|RunnableLambda(get_answers)
    | RunnableLambda(choose_answer)
    )
    with st.chat_message("ai"):
        result = chain.invoke(query)
        st.markdown(result.content)