# src/tools/marketing_rag_tool.py

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
embeddings = OpenAIEmbeddings()
vector_store = None # 벡터 스토어(지식 창고 인덱스)를 저장할 변수

def _initialize_vector_store():
    """
    'marketing_docs' 폴더의 문서를 읽어 벡터 스토어를 초기화합니다.
    """
    global vector_store
    if vector_store is not None:
        print("--- ✅ 지식 창고가 이미 준비되었습니다. ---")
        return

    print("--- 📚 지식 창고(Vector Store)를 구축합니다... ---")
    # 1. 문서 로드
    loader = DirectoryLoader(
        './marketing_docs/',
        glob="**/*[.pdf,.txt]", # PDF와 TXT 파일 모두 로드
        loader_cls=lambda p: PyPDFLoader(p) if p.endswith('.pdf') else TextLoader(p, encoding='utf-8')
    )
    documents = loader.load()

    # 2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 3. 벡터화 및 FAISS 인덱스 생성 (메모리에 저장)
    vector_store = FAISS.from_documents(docs, embeddings)
    print("--- ✅ 지식 창고 구축 완료! ---")

def _rag_marketing_expert(query: str) -> str:
    """
    문서 검색(RAG)을 통해 근거 기반 마케팅 아이디어를 생성합니다.
    """
    print("--- 🧠 RAG 마케팅 전문가 활동 시작 ---")
    
    # 지식 창고가 준비되지 않았다면 먼저 구축
    _initialize_vector_store()

    # 1. (Retrieval) 지식 창고에서 질문과 가장 관련 높은 문서 4개를 검색
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)
    
    # 검색된 문서 내용을 하나의 '컨텍스트'로 합치기
    context = "\n\n---\n\n".join([f"출처: {doc.metadata.get('source', '알 수 없음')}\n\n내용: {doc.page_content}" for doc in retrieved_docs])

    # 2. (Generation) 검색된 컨텍스트를 바탕으로 LLM에게 답변 생성 요청
    prompt = f"""당신은 제공된 [참고 자료]만을 근거로 답변하는 마케팅 전문가입니다.
절대로 당신의 기존 지식을 사용해서는 안 됩니다.

**[사용자 질문]**
{query}

**[참고 자료]**
{context}

**[답변 생성 가이드라인]**
- 위 [참고 자료]를 완벽히 이해하고, 사용자 질문에 대한 답변을 생성하세요.
- 답변의 모든 내용은 반드시 [참고 자료]에 기반해야 합니다.
- 문장 끝에, 어떤 문서를 참고했는지 `[근거: 파일이름.pdf]` 형식으로 명확하게 출처를 밝히세요.

**[최종 답변]**
"""
    
    response = llm.invoke(prompt)
    return response.content

# 최종 도구를 RunnableLambda로 감싸서 builder와 호환되도록 함
marketing_rag_tool = RunnableLambda(_rag_marketing_expert)