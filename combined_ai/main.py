from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

# ===== 공통 및 유틸 =====
import os
import io
import re
import torch
import requests
import httpx
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# ===== 각 기능별 전용 =====
import imagehash                      # 중요도 분석용
from sklearn.cluster import DBSCAN    # 중요도 분석용
import clip                           # CLIP 모델
from transformers import BlipProcessor, BlipForConditionalGeneration  # 캡션 생성
import openai                         # 여행기 생성

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========1차 선별=========== #

# CLIP 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 프롬프트 매핑
prompt_map = {
    "food": "a food-focused travel",
    "activity": "an activity-based travel",
    "nature": "a scenic nature travel",
    "history": "a historical sightseeing travel"
}

# Pydantic 모델 정의
class ImageRequest_select(BaseModel):
    image_id: int
    image_url: str

class AIRequest_select(BaseModel):
    image_list: List[ImageRequest_select]
    purpose: List[str]

class ImportanceResponse(BaseModel):
    image_id: int
    importance: float

# 이미지 선명도 계산
def calculate_sharpness(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"[Sharpness] 다운로드 실패: {image_url} (상태 코드: {response.status_code})")
            return 0.0

        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[Sharpness] 디코딩 실패: {image_url}")
            return 0.0

        lap = cv2.Laplacian(img, cv2.CV_64F)
        return lap.var()
    except Exception as e:
        print(f"[Sharpness] 예외 발생 ({image_url}): {e}")
        return 0.0

# pHash → 이진 벡터 변환
def hash_to_array(h):
    return np.array([int(b) for b in bin(int(str(h), 16))[2:].zfill(64)])

# 이미지 다운로드 (PIL용)
def download_image(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"[Download] 다운로드 실패: {image_url} (상태 코드: {response.status_code})")
            return None
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"[Download] 예외 발생 ({image_url}): {e}")
        return None

# 메인 API 엔드포인트
@app.get("/select-primary-image", response_model=List[ImportanceResponse])
async def analyze_image_list(request: AIRequest_select):
    image_infos = [img.dict() for img in request.image_list]
    user_keywords = request.purpose
    all_keywords = list(prompt_map.keys())
    all_prompts = [prompt_map[k] for k in all_keywords]

    if not user_keywords:
        raise HTTPException(status_code=400, detail="Purpose is required")

    print("[CLIP] 텍스트 임베딩 시작")
    text_tokens = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # pHash 기반 군집화
    hash_vectors = []
    valid_image_infos = []
    for info in image_infos:
        img = download_image(info["image_url"])
        if img is None:
            print(f"[pHash] 이미지 다운로드 실패: {info['image_url']}")
            continue
        img = img.convert("L").resize((32, 32))
        h = imagehash.phash(img)
        hash_vectors.append(hash_to_array(h))
        valid_image_infos.append(info)

    if not hash_vectors:
        raise HTTPException(status_code=422, detail="유효한 이미지가 없습니다.")

    clustering = DBSCAN(metric='hamming', eps=0.27, min_samples=1)
    labels = clustering.fit_predict(hash_vectors)

    grouped = {}
    for label, info in zip(labels, valid_image_infos):
        grouped.setdefault(label, []).append(info)

    # 선택된 대표 이미지의 URL만 따로 set으로 저장
    selected_urls = set()
    for group in grouped.values():
        sharpness_scores = {}
        for info in group:
            score = calculate_sharpness(info["image_url"])
            sharpness_scores[info["image_url"]] = score
            print(f"[Sharpness] {info['image_url']} → {score:.2f}")
        best_path = max(sharpness_scores, key=sharpness_scores.get)
        selected_urls.add(best_path)
        print(f"[Sharpness] 그룹 대표 이미지 선택됨: {best_path}")

    # CLIP 기반 중요도 계산
    result_list = []
    for info in image_infos:
        image_id = info["image_id"]
        image_url = info["image_url"]
        is_selected = image_url in selected_urls

        if is_selected:
            try:
                img = download_image(image_url)
                if img is None:
                    print(f"[CLIP] 이미지 다운로드 실패: {image_url}")
                    importance = 0.0
                else:
                    print(f"[CLIP] 선택된 이미지 처리 중: {image_url}")
                    image = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image)
                        similarity = (image_features @ text_features.T).squeeze(0)
                        probs = similarity.softmax(dim=0).cpu().numpy()

                    user_indices = [all_keywords.index(k) for k in user_keywords if k in all_keywords]
                    user_probs = probs[user_indices]
                    importance = float(np.max(user_probs))
                    print(f"[CLIP] {image_url} 중요도: {importance:.4f}")
            except Exception as e:
                print(f"[CLIP] 처리 실패 ({image_url}): {e}")
                importance = 0.0
        else:
            print(f"[CLIP] 제외된 이미지: {image_url}")
            importance = 0.0

        result_list.append({
            "image_id": image_id,
            "importance": importance
        })

    return result_list
    
# ==========캡셔닝 모델=========== #

# 요청/응답 모델
class ImageRequest_caption(BaseModel):
    image_id: int
    image_url: str

class CaptionRequest(BaseModel):
    image_list: List[ImageRequest_caption]

class CaptionResponse(BaseModel):
    image_id: int
    caption: str

# 모델 핸들러 클래스
class BLIPModelHandler:
    def __init__(self, model_path_or_name: str, use_fp16: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = BlipProcessor.from_pretrained(model_path_or_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_path_or_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" and use_fp16 else torch.float32
        ).to(self.device)


    async def generate_caption_from_url(self, image_url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(image_url)
            if response.status_code != 200:
                raise RuntimeError(f"이미지 요청 실패: {response.status_code}")
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"이미지 다운로드 또는 처리 실패: {e}")

        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

# 모델 인스턴스 생성 (base, finetuned)
model_handler_base = BLIPModelHandler("Salesforce/blip-image-captioning-base")
# 파인튜닝된 모델
finetuned_handler = BLIPModelHandler("/home/gcp_key/Customized_travel_photo_blip")

# 캡션 생성 API
@app.get("/generate-caption", response_model=List[CaptionResponse])
async def generate_caption(req: CaptionRequest):
    results = []
    for image_info in req.image_list:
        try:
            base_caption = await model_handler_base.generate_caption_from_url(image_info.image_url)
            finetuned_caption = await finetuned_handler.generate_caption_from_url(image_info.image_url)
            combined_caption = f"{base_caption}.{finetuned_caption}."
            results.append(CaptionResponse(image_id=image_info.image_id, caption=combined_caption))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{image_info.image_id} 처리 중 오류: {str(e)}")
    return results


# ==========여행기 생성=========== #

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# ====== Pydantic 모델 ======
class ImageRequest_generate(BaseModel):
    image_id: int
    who: Optional[str] = None
    how: Optional[str] = None
    emotion: List[str] = []
    created_at: Optional[str] = None
    location: Optional[str] = None
    style: Optional[str] = None
    caption: Optional[str] = None

class AIRequest_generate(BaseModel):
    image_list: List[ImageRequest_generate]

class ImageDraft(BaseModel):
    image_id: int
    draft: str

class DraftListResponse(BaseModel):
    drafts: List[ImageDraft]


# ====== 여행기 생성 함수 ======
def generate_travel_story(image_list: List[ImageRequest_generate]) -> List[ImageDraft]:
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
    - 전체 흐름은 부드럽게 이어지게 구성
    
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
        max_tokens=80
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
async def generate_travelogue_draft(request: AIRequest_generate):
    return generate_travel_story(image_list=request.image_list)
