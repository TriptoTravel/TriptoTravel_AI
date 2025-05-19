from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import os
import io
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 (운영 시 allow_origins 제한 추천)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class ImageRequest(BaseModel):
    image_id: int
    image_url: str

class CaptionRequest(BaseModel):
    image_list: List[ImageRequest]

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


    def generate_caption_from_url(self, image_url: str) -> str:
        try:
            response = requests.get(image_url, timeout=5)
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
def generate_caption(req: CaptionRequest):
    results = []
    for image_info in req.image_list:
        try:
            base_caption = model_handler_base.generate_caption_from_url(image_info.image_url)
            finetuned_caption = finetuned_handler.generate_caption_from_url(image_info.image_url)
            combined_caption = f"{base_caption}.{finetuned_caption}."
            results.append(CaptionResponse(image_id=image_info.image_id, caption=combined_caption))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{image_info.image_id} 처리 중 오류: {str(e)}")
    return results
