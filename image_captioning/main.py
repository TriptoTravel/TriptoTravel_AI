from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import os
import io
from fastapi.middleware.cors import CORSMiddleware
import torch 

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 핸들러 클래스
class BLIPModelHandler:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        # GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_caption_from_url(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"이미지 다운로드 또는 처리 실패: {e}")

        inputs = self.processor(image, return_tensors="pt").to(self.device)  # 입력 데이터를 GPU로 이동
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

# Pydantic 모델
class ImageRequest(BaseModel):
    image_id: int
    uri: str  # GCS 등 외부 이미지 URL

class CaptionRequest(BaseModel):
    images: List[ImageRequest]

class CaptionResponse(BaseModel):
    image_id: int
    draft: str

# BLIP 모델 초기화
model_handler = BLIPModelHandler()

# 캡션 생성 API
@app.post("/generate-caption", response_model=List[CaptionResponse])
def generate_caption(req: CaptionRequest):
    results = []
    for image_info in req.images:
        try:
            caption = model_handler.generate_caption_from_url(image_info.uri)
            results.append({"image_id": image_info.image_id, "draft": caption})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{image_info.image_id} 처리 중 오류: {str(e)}")

    return results
