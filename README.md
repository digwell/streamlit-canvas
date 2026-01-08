# 이미지 분류기

`huggingface`의 `google/vit-base-patch16-224` 로

업로드한 이미지를 분류한다.

필요시 모델 추가 가능하다.

`ollama` 가 설치되어 있다면 `gemma3:12b` 사용해 보길 추천한다.

`ollama` 에는 score 부분은 사용하지 않는다.

.env

```.env
OLLAMA_HOST=192.168.1.2
OLLAMA_PORT=11434
```

필요 라이브러리 설치

```shell
pip install -r requirements.txt
```

실행

```shell
streamlit run app.py
```
