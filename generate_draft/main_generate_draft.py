from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import re

# FastAPI 애플리케이션 객체 생성
app = FastAPI()


# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# ====== Pydantic 모델 ======
class ImageRequest(BaseModel):
    image_id: int
    who: Optional[str] = None
    how: Optional[str] = None
    emotion: List[str] = []
    created_at: Optional[str] = None
    location: Optional[str] = None
    style: Optional[str] = None
    caption: Optional[str] = None

class AIRequest(BaseModel):
    image_list: List[ImageRequest]

class ImageDraft(BaseModel):
    image_id: int
    draft: str

class DraftListResponse(BaseModel):
    drafts: List[ImageDraft]


# ====== 여행기 생성 함수 ======
def generate_travel_story(image_list: List[ImageRequest]) -> List[ImageDraft]:
    style = image_list[0].style or "요약형"
    image_infos = []

    for idx, image in enumerate(image_list):
        caption = image.caption or ""
        caption_split = caption.split('.')
        sentence_1 = caption_split[0].strip() if len(caption_split) > 0 else ""
        sentence_2 = caption_split[1].strip() if len(caption_split) > 1 else ""

        info = f"""
### IMAGE {idx+1} ###
- 날짜: {image.created_at or '미상'}
- 장소: {image.location or '미상'}
- 구성원: {image.who or '미상'}
- 활동 요약: {caption}
- 가장 인상 깊었던 일: {image.how or '없음'}
- 감정: {', '.join(image.emotion) if image.emotion else '없음'}

주어진 두 문장 중 장소, 가장 인상 깊었던 일에 적합한 문장을 선택하고, 두 문장 다 적합하다면 두 문장을 자연스럽게 조합하여 여행기에 반영해주세요:
1. {sentence_1}
2. {sentence_2}
예를 들어:
    - 문장 1: "음식이 놓은 테이블이 있고, 그 위에 음식이 가득 놓여있어요."
    - 문장 2: "치즈와 소스를 얹은 피자."
    - 결합된 문장: "테이블 위에 치즈와 소스를 얹은 피자가 가득 놓여있어요."

이 활동, 감정이 여행기에 자연스럽게 반영되었는지 확인해 주세요:
- 활동 요약: {caption}
- 인상 깊은 일: {image.how or '없음'}
- 감정: {', '.join(image.emotion) if image.emotion else '없음'}
""".strip()
        image_infos.append(info)

    image_summary = "\n\n".join(image_infos)

    # 스타일 분기
    if style == "정보형":
        style_instruction = """
    정보형 여행기의 특징:
    - 목적: 여행지의 정확한 정보와 팁을 독자에게 전달합니다.
    - 문체: '~입니다', '~합니다' 형태의 존댓말 사용
    - 감정 표현은 최소화하고, 사실 위주로 씁니다.
    - 시간과 장소를 반드시 포함하세요:
        - 시간은 'YYYY년 MM월 DD일 오전/오후 HH시 mm분' 형식
        - 장소는 도로명 주소 전체를 문장 속에 자연스럽게 녹여주세요
        
    예시:
    - 2024년 3월 21일 오전 6시 20분, 제주특별자치도 서귀포시 성산읍 일출로 284-12에 위치한 성산일출봉에 올랐습니다. 이곳은 유네스코 세계자연유산으로 지정된 분화구로, 일출을 보기 위해 많은 관광객이 찾는 명소입니다.

    """
    elif style == "요약형":
        style_instruction = """
    요약형 여행기의 특징:
    - 목적: 주요 활동과 감정을 간결하게 정리
    - 문체: '~이다', '~였다', '~하지 않았다' 형식의 중립적 서술체 사용
    - 구성: 각 활동을 2~3문장 내외로 요약
    - 감정 표현은 유지하되 정보는 간단하게
    
    예시:
    - 오후 늦게 도착한 성산일출봉은 여전히 많은 사람들로 붐볐다. 우리는 잠시 전망대에 올라 맑은 하늘과 바다를 바라보았다.
    """
    else:
        style_instruction = """
    감성형 여행기의 특징:
    - 문체: '~이다', '~였다' 같은 중립적 서술체
    - 정보보다 감정과 분위기에 집중
    - 이미지 설명은 절대 따로 쓰지 말고, 감정 속에 자연스럽게 녹여 쓸 것
    - 전체 문장은 부드럽게 이어져야 하며, 은유적 표현을 포함
    예시:
    - 아침 햇살에 비친 협재해변은 조용하고 따뜻했다. 복잡했던 마음이 파도 소리와 함께 떠내려간 듯 잔잔해졌다.
    """

    # 전체 프롬프트 생성
    prompt = f"""
당신은 20~30대 인기 여행 블로거입니다.

아래는 여행 중 찍은 여러 장의 사진에 대한 설명입니다.  
각 설명을 바탕으로 각각의 여행기 단락을 작성해 주세요.  
블로그나 일기처럼 전체적인 내용이 연속적으로 이어지도록 써주세요.

- 각 단락 앞에는 반드시 `### IMAGE 1 ###`, `### IMAGE 2 ###` 형식의 마커를 붙이세요.
- 이 마커는 오직 **여행기 본문에만** 붙이고, 아래 정보 요약은 단지 참고용입니다.
- 마커 이후에는 해당 이미지에 대한 3~5문장 분량의 자연스러운 여행기 단락을 작성하세요.
- 모든 단락이 잘 이어져서 하나의 여행기처럼 보이게 작성해주세요.
- 문체는 스타일에 따라 아래 규칙을 따르세요.
- caption을 통해 상황과 분위기만 유추하고, 전반적인 일기 내용은 how로 구성할 것
- 절대 caption 내용을 그대로 넣지말 것

{style_instruction}

다음은 이미지 요약 정보입니다:

{image_summary}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful travel blogger."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    full_text = response['choices'][0]['message']['content'].strip()

    # === IMAGE 블록을 파싱 ===
    pattern = r"### IMAGE (\d+) ###\s*(.*?)(?=### IMAGE \d+ ###|$)"
    matches = re.findall(pattern, full_text, re.DOTALL)

    draft_list = []
    for match in matches:
        idx = int(match[0]) - 1
        draft_text = match[1].strip()
        if 0 <= idx < len(image_list):
            draft_list.append(ImageDraft(
                image_id=image_list[idx].image_id,
                draft=draft_text
            ))

    return draft_list

# ====== API 엔드포인트 ======
@app.get("/generate-travel-log", response_model=List[ImageDraft])
async def generate_travelogue_draft(request: AIRequest):
    return generate_travel_story(image_list=request.image_list)
