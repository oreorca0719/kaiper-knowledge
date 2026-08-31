"""
doc_topic Classifier — 문서 전체를 LLM이 1회 보고 주제 분류.

호출 빈도: 문서당 1회 (인제스트 시점)
용도: retriever가 query의 topic과 chunk doc_topic 매칭하여 score boost
     같은 entity를 공유하지만 주제가 다른 문서들 (예: 01 vs 02 PPT) 분리
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.core.config import get_llm
from app.core.history_utils import extract_text_content


_DOC_TOPIC_PROMPT = """다음 문서의 주제를 한 단어 또는 짧은 구절로 분류하세요.

권장 라벨 (자유롭게 새 라벨도 허용):
- branding (브랜드 정체성·슬로건·가치)
- history (회사 연혁·사업 구조 변화)
- competitor_analysis (경쟁사 비교·시장)
- education_plan (교육 과정·커리큘럼 기획)
- vendor_report (벤더·플랫폼 도입 보고)
- meeting_minutes (회의록·논의 결과)
- feedback (피드백·검토 의견)
- instructor_sourcing (강사 모집·후보)
- cost_structure (비용·예산·견적)
- ai_model_inquiry (AI 모델 사용·요금제 문의)
- spec_doc (시스템·기능 사양)

【출력】
한 단어 또는 짧은 영문 snake_case 구절. 다른 텍스트 금지.
예: branding, vendor_report, education_plan_curriculum

[문서 앞부분 3000자]
{snippet}
"""


def classify_doc_topic(doc_text: str, model_name: str | None = None) -> str:
    """문서 전체 텍스트를 받아 doc_topic 분류 (LLM 1회).

    실패 시 'general' 반환 (graceful degradation).
    """
    if not doc_text or not doc_text.strip():
        return "general"
    snippet = doc_text[:3000]
    try:
        resp = get_llm(model_name).invoke([
            HumanMessage(content=_DOC_TOPIC_PROMPT.format(snippet=snippet))
        ])
        raw = extract_text_content(resp.content)
        topic = raw.strip().split("\n")[0].strip()
        # 정규화: 영문 소문자 + 언더스코어만
        topic = "".join(c if (c.isalnum() or c in "_-") else "_" for c in topic.lower())
        topic = topic.strip("_-")
        if not topic or len(topic) > 40:
            return "general"
        return topic
    except Exception as e:
        print(f"[DOC_TOPIC] classify failed (non-fatal): {e}")
        return "general"
