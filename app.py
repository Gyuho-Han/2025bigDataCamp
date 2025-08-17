# app.py
import streamlit as st
import os, json, requests
from dotenv import load_dotenv

# LangChain / OpenAI imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1) 환경 변수 로드
load_dotenv()

# 2) JSON → Document 리스트 로드
@st.cache_data(show_spinner=False)
def load_documents(path="seoul_tour_clean.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for item in raw:
        content = f"{item['명칭']}\n{item.get('개요','')}\n{item.get('상세정보','')}"
        metadata = {k: item.get(k, "") for k in ["주소","문의","쉬는날","이용시간","주차시설","명칭"]}
        docs.append(Document(page_content=content.strip(), metadata=metadata))
    return docs

# 3) VectorStore 초기화 (_docs 언더바 처리)
@st.cache_resource(show_spinner=False)
def init_vectorstore(_docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini", chunk_size=250, chunk_overlap=25
    )
    chunks = splitter.split_documents(_docs)
    embed = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory="./chroma_db"
    )
    db.persist()
    return db

# 4) Hybrid Retriever 초기화 (_docs, _vector_db 언더바 처리)
@st.cache_resource(show_spinner=False)
def init_retriever(_docs, _vector_db):
    bm25 = BM25Retriever.from_documents(_docs)
    bm25.k = 15
    vect = _vector_db.as_retriever(search_kwargs={"k": 15})
    return EnsembleRetriever(retrievers=[bm25, vect], weights=[0.5, 0.5])

# 5) Reranking Retriever 초기화 (_hybrid)
@st.cache_resource(show_spinner=False)
def init_rerank_retriever(_hybrid):
    # 재순위용 LLM (약간 높은 온도)
    llm_rerank = AzureChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    re_ranker = LLMListwiseRerank.from_llm(
        llm=llm_rerank, top_n=10, verbose=True
    )
    return ContextualCompressionRetriever(
        base_compressor=re_ranker,
        base_retriever=_hybrid
    )

# 6) 날씨 도구
@tool
def get_weather() -> str:
    """서울 현재 날씨와 강수 여부를 반환합니다."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return "ERROR: OPENWEATHER_API_KEY missing"
    data = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?"
        f"q=Seoul&appid={key}&units=metric&lang=kr"
    ).json()
    desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    rain = "☔️ 비 오는 중" if "Rain" in [w["main"] for w in data["weather"]] else "☀️ 비 안 옴"
    return json.dumps({"날씨": desc, "기온": f"{temp}°C", "강수": rain}, ensure_ascii=False)

# 7) RAG+Rerank 검색 도구
@tool
def search_tour_place(query: str) -> str:
    """질문에 맞는 서울 관광지 청크를 reranking 후 반환합니다."""
    docs = st.session_state.rerank_retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

# 8) AgentExecutor 초기화 (원본 상세 프롬프트 포함)
@st.cache_resource(show_spinner=False)
def init_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 서울 토박이 관광 전문가 AI입니다. 도구를 적절히 사용하여 사용자의 요청을 해결하세요.  
가능한 한 정확하고 간결하게 답하며, 질문이 명확하지 않다면 정중히 되묻습니다.  
항상 도구를 사용하여 답하세요. 날씨가 좋다면 야외로, 날씨가 좋지 않으면 실내로 추천해주세요.  
현재 서울의 날씨도 설명해주세요. tool의 정보를 재구성해 설명해주세요.

[지침]
- 컨텍스트에 있는 정보만을 사용하여 답변할 것  
- 외부 지식이나 정보를 사용하지 말 것  
- 질문이 불명확하면 정중하게 되묻습니다.  
- 도구 사용 결과는 복붙하지 말고, 자연스러운 문장으로 재구성하세요.  
- 추천 장소는 정확한 명칭과 함께 설명합니다.  
- 추천 이유와 선정 기준을 명확히 밝히세요. (예: 실외 활동, 역사적 가치, 가족 단위 등)  
- 날씨를 고려해서 장소를 선정하시오.  
- 장소는 5곳 선정하시오  
- 컨텍스트에서 답을 찾을 수 없는 경우 "죄송하지만 그런 장소는 서울에 없는 것 같습니다."라고 응답할 것  
- 답변은 논리적이고 구조화된 형태로 제공할 것  
- 답변은 한국어를 사용할 것  
- 서울에 오래산 사람이 지방에서 올라온 사람에게 여행지를 추천해주는 느낌으로 대답할 것  
"""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = AzureChatOpenAI(model="gpt-4.1", temperature=0)
    tools = [get_weather, search_tour_place]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# —— Streamlit 앱 시작 —— 
st.set_page_config(page_title="서울 관광지 추천 AI", page_icon="🗺️")
st.title("🗺️ 서울 관광지 추천 AI")

# 리소스 초기화
docs = load_documents()
vector_db = init_vectorstore(docs)
st.session_state.hybrid_retriever = init_retriever(docs, vector_db)
st.session_state.rerank_retriever = init_rerank_retriever(st.session_state.hybrid_retriever)
agent_executor = init_agent()

# 사용자 입력 UI
query = st.text_input(
    "여행 목적/상황을 입력하세요", 
    placeholder="예: 비 오는 날 가족과 갈만한 곳 추천해줘"
)

if st.button("추천받기") and query:
    with st.spinner("추천 생성 중..."):
        output = agent_executor.invoke({"input": query})["output"]
    st.markdown("**📝 추천 결과**")
    st.write(output)
