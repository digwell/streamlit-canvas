import base64
import json
from io import BytesIO

import requests
from PIL import Image
from transformers import pipeline


class MyModel:
    def __init__(self):
        pass

    def ollama_predict(self, model, image_data):
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
        prompt = "어떤 이미지인지 알려줘. 단답형으로."

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
            response = requests.post(
                endpoint, headers=headers, data=json.dumps(data), timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                # Ollama 응답에서 결과 추출 (텍스트 중 숫자(0~9)만)
                output = result.get("response", "").strip()

                return output
            else:
                return f"ollama 요청 실패: {response.status_code} - {response.text}"
        except Exception as e:
            return f"ollama 요청 중 예외: {e}"

    def huggingface_predict(self, model, image_data):
        # huggingface 모델을 이용하여 이미지 분류 예측을 수행
        from PIL import Image

        # RGB 변환
        if image_data.shape[2] == 4:
            pil_img = Image.fromarray(image_data.astype("uint8")).convert("RGB")
        else:
            pil_img = Image.fromarray(image_data.astype("uint8"))

        # 이미지를 224x224 크기로 리사이즈
        pil_img = pil_img.resize((224, 224))

        try:
            vqa_pipeline = pipeline("image-classification", model=model)
            result = vqa_pipeline(pil_img)

        except Exception as e:
            return {"error": f"모델 실행 중 오류: {e}"}

        # image-classification 결과 처리 (리스트 형태: [{"label": "...", "score": ...}, ...])
        if isinstance(result, list) and len(result) > 0:
            # 상위 결과 반환z
            top_result = result[0]
            return {
                "label": top_result.get("label", "unknown"),
                "score": top_result.get("score", 0.0),
                "all_results": result,  # 전체 결과도 포함
            }
        # 다른 형태의 결과 처리
        elif isinstance(result, dict):
            if "label" in result and "score" in result:
                return result
            elif "answer" in result:
                return {"label": result.get("answer", "unknown"), "score": 1.0}
            else:
                return {"label": str(result), "score": 1.0}
        elif isinstance(result, str):
            return {"label": result, "score": 1.0}
        else:
            return {"label": str(result), "score": 1.0}

    def predict_image(self, selected_model, image_data):
        ai_service = selected_model.split(" - ")[0]
        model = selected_model.split(" - ")[1]

        if ai_service == "ollama":
            text = self.ollama_predict(model, image_data)
            return {"text": text}

        elif ai_service == "huggingface":
            result = self.huggingface_predict(model, image_data)
            # 결과가 딕셔너리 형태면 그대로 반환, 아니면 텍스트로 변환
            if isinstance(result, dict):
                return result
            else:
                return {"text": str(result)}
        else:
            return "unknown"
