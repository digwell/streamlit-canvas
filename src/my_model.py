import base64
import json
import random
import re
from datetime import datetime
from io import BytesIO

import numpy as np
import onnx
import onnxruntime as ort
import requests
import torch
from PIL import Image
from torchvision import datasets, transforms
from transformers import pipeline


class MyModel:
    def __init__(self):
        pass

    def get_prompt(self,ai_type):
        if ai_type == "digit":
            return (
                "아래 이미지는 사람이 손글씨로 쓴 숫자입니다. 이미지를 보고 숫자가 무엇인지 한 자리 숫자(0~9)로만 답변해 주세요. "
                "숫자만 답변해 주세요.\n\n"
            )
        elif ai_type == "alphabet":
            return (
                "아래 이미지는 사람이 손글씨로 쓴 알파벳 중 하나입니다. 이미지를 보고 영문 알파벳이 대소문자 구분하여 무엇인지 답변해 주세요."
                "알파벳만 답변해 주세요.\n\n"
            )
        elif ai_type == "hangul":
            return (
                "아래 이미지는 사람이 손글씨로 쓴 한글입니다. 이미지를 보고 한글이 무엇인지 답변해 주세요."
                "설명이 필요 없이 한글 한글자 만 답변해 주세요.\n\n"
            )
        else:
            return None


    def ollama_predict(self, ai_type, model, image_data):

        # 이미지 배열(image_data)을 base64로 인코딩하여 ollama API로 보내고 응답의 숫자만 추출

        # 1. image_data (numpy array, RGBA) -> PIL Image -> RGB -> bytes -> base64
        if image_data.shape[2] == 4:
            pil_img = Image.fromarray(image_data.astype("uint8")).convert("RGB")
        else:
            pil_img = Image.fromarray(image_data.astype("uint8"))


        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode()

        # 2. 자연어 프롬프트 준비
        prompt = self.get_prompt(ai_type)

        # 3. ollama API 호출 (예: http://localhost:11434/api/generate)
        endpoint = "http://192.168.219.17:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
        }

        try:
            response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=60)
            if response.status_code == 200:
                result = response.json()
                print(result)
                # Ollama 응답에서 결과 추출 (텍스트 중 숫자(0~9)만)
                output = result.get("response", "").strip()
                if ai_type == "digit":
                    match = re.search(r"\b[0-9]\b", output)
                elif ai_type == "alphabet":
                    match = re.search(r"\b[a-zA-Z]\b", output)
                elif ai_type == "hangul":
                    match = re.search(r"\b[ㄱ-ㅎㅏ-ㅣ가-힣]\b", output)

                if match:
                    return match.group(0)
                else:
                    # 숫자 직접 못 찾으면 원문 출력
                    return output
            else:
                return f"ollama 요청 실패: {response.status_code} - {response.text}"
        except Exception as e:
            return f"ollama 요청 중 예외: {e}"


    def huggingface_predict(self, ai_type, model, image_data):
        # huggingface 모델을 이용하여 이미지 분류 예측을 수행
        # ai_type으로 프롬프트를 구하고 pipeline에 연결
        import numpy as np
        from PIL import Image

        # RGB 변환
        if image_data.shape[2] == 4:
            pil_img = Image.fromarray(image_data.astype("uint8")).convert("RGB")
        else:
            pil_img = Image.fromarray(image_data.astype("uint8"))

        # ai_type으로 프롬프트 구하기
        prompt = self.get_prompt(ai_type)

        # transformers pipeline 사용 (vision-language 모델 사용)
        # 프롬프트와 이미지를 함께 전달
        vqa_pipeline = pipeline("image-text-to-text", model=model)
        result = vqa_pipeline(pil_img, prompt)

        # 결과에서 텍스트 추출
        label = result if isinstance(result, str) else result.get("answer", "unknown") if isinstance(result, dict) else str(result)

        return label


    def save_image(self, image_data):
        image = Image.fromarray(image_data.astype("uint8"))
        uid = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = f"images/{uid}.png"
        image.save(image_path)
        return image_path


    def predict_image(self, ai_type, selected_model, image_data):
        ai_service = selected_model.split(" - ")[0]
        model = selected_model.split(" - ")[1]

        image_path = self.save_image(image_data)

        if ai_service == "ollama":
            number = self.ollama_predict(ai_type, model, image_data)
            return {"text": number, "image_path": image_path}

        elif ai_service == "huggingface":
            label = self.huggingface_predict(ai_type, model, image_data)
            return {"text": label, "image_path": image_path}
        else:
            return "unknown"
