# src/graph/builder.py

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# 새로운 상태와 우리가 만든 모든 '도구(Tool)'들을 가져옵니다.
from graph.state import AgentState
from tools.data_analysis_tool import data_analysis_tool
from tools.web_search_tool import web_search_tool
from tools.api_call_tool import api_caller_tool
from tools.marketing_rag_tool import marketing_rag_tool

# LLM 초기화
load_dotenv() # .env 파일에서 API 환경 변수 로드
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# --- 1. 도구(Tool) 등록 ---
# 모든 도구를 이름과 함께 딕셔너리로 묶어 Executor가 쉽게 찾아 쓸 수 있도록 합니다.
tools = {
    "data_analyzer": data_analysis_tool,
    "web_searcher": web_search_tool,
    "api_caller": api_caller_tool,
    "marketing_expert": marketing_rag_tool,
}

# --- 2. 챗봇의 새로운 두뇌: 노드(Node) 정의 ---

def planner_node(state: AgentState):
    """상권 특성을 먼저 분석하는 '전략 컨설턴트'로서 계획을 수립합니다."""
    print("--- 🤔 전략 컨설턴트(Planner) 활동 시작 ---")
    
    prompt = f"""당신은 데이터 기반으로 소상공인에게 맞춤형 솔루션을 제공하는 AI 전략 컨설턴트입니다.
당신은 반드시 아래의 **[전략 컨설팅 프레임워크]**에 따라 사고하고, 실행 가능한 계획을 생성해야 합니다.

**[전략 컨설팅 프레임워크]**
1.  **1단계 (상권 분석):** 가장 먼저, `data_analyzer`를 사용해 해당 가맹점이 속한 상권의 인구통계학적 특성(주요 연령대, 성별), 유동인구 특성(주중/주말, 시간대별), 주요 업종 등을 분석하여 상권의 유형(예: 대학가, 오피스, 주거)을 정의합니다.
2.  **2단계 (내부 문제 진단):** 1단계에서 파악한 상권 특성을 바탕으로, 가맹점의 내부 데이터(매출, 고객, 재방문율 등)를 분석하여 '상권 특성과 맞지 않는' 문제점을 진단합니다.
3.  **3단계 (솔루션 수립):** 진단된 문제점을 해결하기 위해 `web_searcher`로 성공 사례를 찾거나, `marketing_expert`로 상권 특성에 맞는 창의적인 아이디어를 도출합니다.

**[매우 중요한 출력 규칙]**
- 당신의 최종 출력물은 **오직 `[Tool: 도구이름]` 형식으로 시작하는 번호 매겨진 실행 계획 목록**이어야 합니다.
- 절대 서론, 요약 등 다른 설명을 포함하지 마세요.

**사용 가능한 도구:**
- **data_analyzer**: 데이터 파일을 심층 분석합니다.
- **web_searcher**: 최신 트렌드, 성공 사례 등을 검색합니다.
- **marketing_expert**: 'marketing_docs'의 전문 지식을 기반으로 마케팅 전략을 생성합니다.

**사용자 요청:** "{state['messages'][-1].content}"

**위 프레임워크에 따라, 실행 계획 목록만 생성해주세요:**
"""
    
    response = llm.invoke(prompt)
    
    # (핵심 수정!) LLM의 출력물에서 '[Tool:'로 시작하는 줄만 '계획'으로 인정합니다.
    plan = [
        step.strip() for step in response.content.split('\n') 
        if step.strip() and '[Tool:' in step
    ]
    
    print(f"--- 📝 수립된 최종 계획 ---\n" + "\n".join(plan))
    return {"plan": plan, "past_steps": []}

def executor_node(state: AgentState):
    """계획의 다음 단계를 실행하는 '실행 전문가'"""
    print("--- ⚙️ 계획 실행(Executor) 시작 ---")
    
    step = state["plan"][0]
    
    try:
        tool_name = step.split("[Tool: ")[1].split("]")[0]
        query = step.split("]")[1].strip()
    except IndexError:
        return {"past_steps": state.get("past_steps", []) + [(step, "오류: 계획 형식이 잘못되었습니다.")]}

    print(f"---  [실행] 도구: {tool_name} // 질문: {query} ---")
    
    if tool_name in tools:
        tool = tools[tool_name]
        try:
            result = tool.invoke(query)
            past_step = (step, str(result))
        except Exception as e:
            # (핵심 수정!) 잡힌 오류의 내용을 터미널에 자세히 출력합니다.
            print(f"--- 🚨 EXECUTOR가 도구 실행 중 오류 감지: {e} ---") 
            past_step = (step, f"도구 실행 중 오류 발생: {e}")
            
        return {
            "plan": state["plan"][1:],
            "past_steps": state.get("past_steps", []) + [past_step]
        }
    else:
        return {"past_steps": state.get("past_steps", []) + [(step, "오류: 알 수 없는 도구입니다.")]}

def synthesizer_node(state: AgentState):
    """수집된 '구체적인 근거'를 인용하여 최종 컨설팅 보고서를 작성합니다."""
    print("--- ✍️ 데이터 기반 컨설턴트(Synthesizer) 최종 보고서 작성 ---")

    evidence = "\n\n".join(
        [f"**실행 계획:** {step}\n**수집된 근거:**\n{result}" for step, result in state.get("past_steps", [])]
    )
    
    prompt = f"""당신은 수집된 모든 근거 자료를 종합하여, 소상공인을 위한 최종 비즈니스 컨설팅 보고서를 작성하는 시니어 컨설턴트입니다.

**[매우 중요한 보고서 작성 규칙]**
- **모든 주장은 반드시 [수집된 근거 자료]에서 찾은 구체적인 수치나 사실을 직접 인용해야 합니다.**
- '두루술하게' 표현하지 말고, 정확한 데이터를 제시하여 주장의 설득력을 높여야 합니다.

**[근거 제시 예시]**
- **나쁜 예시:** 고객층이 다양하지 않습니다. [근거: 데이터 분석 결과]
- **좋은 예시:** **고객의 85%가 20대이며, 특히 '대학생' 고객 비중이 절대적**으로 나타나 고객층이 매우 편중되어 있습니다. [근거: big_data_set2_f.csv '연령대' 및 '직업' 컬럼 분석 결과]

**[사용자의 초기 질문]**
{state['messages'][0].content}

**[수집된 근거 자료]**
{evidence}

**[보고서 형식]**
### 📝 문제점 진단 (구체적인 데이터를 근거로 제시)
(예: '데이터 분석 결과, 주말 매출이 주중 대비 60%나 급감하는 것으로 나타났습니다...')

### 💡 해결 방안 제안 (상권 특성과 문제점을 연결하여 제시)
(예: '주말 유동인구가 적은 오피스 상권의 특성을 고려하여, 주중 점심시간 직장인 대상 구독 서비스를 제안합니다...')

### 📚 핵심 근거 자료 (각 주장의 출처 명시)
(예: 주말 매출 감소 [근거: big_data_set1_f.csv '요일별_매출' 분석 결과])

**위 규칙과 예시를 반드시 준수하여, 데이터 기반의 최종 컨설팅 보고서를 작성해주세요:**
"""

    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}

# --- 3. 새로운 그래프 구성 ---

workflow = StateGraph(AgentState)

# 새로운 전문가 노드들을 그래프에 추가
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

# 시작점은 항상 'planner'
workflow.set_entry_point("planner")

# 각 노드를 연결
workflow.add_edge("planner", "executor")
workflow.add_edge("synthesizer", END)

# Executor는 조건부로 연결: 실행할 계획이 남았는지 확인
def should_continue(state: AgentState):
    if state.get("plan"):
        return "executor" # 아직 실행할 계획이 남았으면 executor로 다시 이동 (루프)
    else:
        return "synthesizer" # 계획이 모두 끝났으면 synthesizer로 이동

workflow.add_conditional_edges("executor", should_continue)

# 그래프 최종 컴파일
graph = workflow.compile(checkpointer=MemorySaver())
print("=========================================//")
print("✅ '계획-실행-종합' 모델로 LangGraph가 성공적으로 컴파일되었습니다!")
# print(">> 그래프 노드:", graph.nodes)
print(">> 그래프 도구 목록:", list(tools.keys()))
print("=========================================")
print("시작 >>")