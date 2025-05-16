from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import imagehash
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import torch
import clip
import requests
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
class ImageRequest(BaseModel):
    image_id: int
    image_url: str

class AIRequest(BaseModel):
    images: List[ImageRequest]
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
async def analyze_images(request: AIRequest):
    image_infos = [img.dict() for img in request.images]
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

    selected_infos = []
    for group in grouped.values():
        sharpness_scores = {}
        for info in group:
            score = calculate_sharpness(info["image_url"])
            sharpness_scores[info["image_url"]] = score
            print(f"[Sharpness] {info['image_url']} → {score:.2f}")
        best_path = max(sharpness_scores, key=sharpness_scores.get)
        for info in group:
            if info["image_url"] == best_path:
                selected_infos.append(info)
                print(f"[Sharpness] 그룹 대표 이미지 선택됨: {info['image_url']}")

    # CLIP 기반 중요도 계산
    result_list = []
    for info in image_infos:
        image_id = info["image_id"]
        image_url = info["image_url"]
        is_selected = any(sel["image_id"] == image_id for sel in selected_infos)

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
