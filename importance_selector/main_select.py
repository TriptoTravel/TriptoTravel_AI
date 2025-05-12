from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from PIL import Image
import imagehash
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import torch
import clip
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import requests
from io import BytesIO

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CLIP 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 프롬프트 매핑 (자연어 목적 → 영어 프롬프트)
prompt_map = {
    "food": "a food-focused travel",
    "activity": "an activity-based travel",
    "nature": "a scenic nature travel",
    "history": "a historical sightseeing travel"
}

# === Pydantic 모델 ===
class ImageRequest(BaseModel):
    image_id: int
    uri: str  # 실제로는 GCS 경로

class AIRequest(BaseModel):
    images: List[ImageRequest]
    purpose: List[str]

class ImportanceResponse(BaseModel):
    image_id: int
    importance: float

# === 이미지 sharpness 계산 ===
def calculate_sharpness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()

# === pHash → 이진벡터 변환 ===
def hash_to_array(h):
    return np.array([int(b) for b in bin(int(str(h), 16))[2:].zfill(64)])

@app.post("/select-primary-image", response_model=List[ImportanceResponse])
async def analyze_images(request: AIRequest):
    image_infos = [img.dict() for img in request.images]
    user_keywords = request.purpose
    all_keywords = list(prompt_map.keys())
    all_prompts = [prompt_map[k] for k in all_keywords]

    if not user_keywords:
        return []

    # 텍스트 임베딩 (전체 키워드 기준)
    text_tokens = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # === pHash 버퍼생성 ===
    hash_vectors = []
    for info in image_infos:
        response = requests.get(info["uri"])
        img = Image.open(BytesIO(response.content)).convert("L").resize((32, 32))

        
        h = imagehash.phash(img)
        hash_vectors.append(hash_to_array(h))

    clustering = DBSCAN(metric='hamming', eps=0.27, min_samples=1)
    labels = clustering.fit_predict(hash_vectors)

    grouped = {}
    for label, info in zip(labels, image_infos):
        grouped.setdefault(label, []).append(info)

    selected_infos = []
    for group in grouped.values():
        sharpness_scores = {info["uri"]: calculate_sharpness(info["uri"]) for info in group}
        best_path = max(sharpness_scores, key=sharpness_scores.get)
        for info in group:
            if info["uri"] == best_path:
                selected_infos.append(info)

    # === 중요도 계산 ===
    result_list = []
    for info in image_infos:
        image_id = info["image_id"]
        uri = info["uri"]
        is_selected = any(sel["image_id"] == image_id for sel in selected_infos)

        if is_selected:
            try:
                response = requests.get(uri)
                image = preprocess(Image.open(BytesIO(response.content))).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image)
                    similarity = (image_features @ text_features.T).squeeze(0)
                    probs = similarity.softmax(dim=0).cpu().numpy()

                user_indices = [all_keywords.index(k) for k in user_keywords if k in all_keywords]
                user_probs = probs[user_indices]
                importance = float(np.max(user_probs))
            except Exception as e:
                print(f"{uri} 처리 실패: {e}")
                importance = 0.0
        else:
            importance = 0.0

        result_list.append({
            "image_id": image_id,
            "importance": importance
        })

    return result_list
