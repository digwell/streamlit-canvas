import os
import socket
from datetime import datetime

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.my_model import MyModel

load_dotenv()


@st.cache_resource
def get_model_instance():
    return MyModel()


# AI 머신의 ollama 포트가 열려있는지 확인
def is_ollama_port_open():
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    OLLAMA_PORT = os.getenv("OLLAMA_PORT")

    print(OLLAMA_PORT)

    if OLLAMA_HOST is None or OLLAMA_PORT is None:
        return False
    try:
        with socket.create_connection(("192.168.219.17", 11434), timeout=1):
            return True
    except Exception:
        return False


st.set_page_config(page_title="미션15", page_icon="🚀")

st.title("필기 인식")
st.write("이미지를 업로드하여 숫자나 문자를 인식하세요.")

with st.form("form1"):
    uploaded_file = st.file_uploader(
        "이미지 파일을 선택하세요",
        type=["png", "jpg", "jpeg"],
        help="PNG, JPG, JPEG 형식의 이미지를 업로드할 수 있습니다.",
    )

    if uploaded_file is not None:
        # 업로드된 이미지 미리보기
        img_preview = Image.open(uploaded_file)
        st.image(img_preview, caption="업로드된 이미지", width=224)

    models = []

    if is_ollama_port_open():
        models.append("ollama - gemma3:12b")

    models.append("huggingface - google/vit-base-patch16-224")

    selected_model = st.radio("모델 선택", models)

    submit_button = st.form_submit_button("Submit")


if submit_button:
    if uploaded_file is None:
        st.error("이미지를 업로드해주세요.")
        st.stop()

    my_model = get_model_instance()

    # 업로드된 이미지를 PIL Image로 열고 numpy array로 변환
    img = Image.open(uploaded_file)
    # RGB로 변환 (RGBA인 경우)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # PIL Image를 numpy array로 변환
    image_data = np.array(img)

    predict_image = my_model.predict_image(selected_model, image_data)

    # 날짜+시각
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 이미지 (작게 출력)
    img_thumbnail = img.copy()
    img_thumbnail.thumbnail((64, 64))

    # 결과 처리 및 표시
    if (
        isinstance(predict_image, dict)
        and "label" in predict_image
        and "score" in predict_image
    ):
        # 분류 모델 결과 (label, score)
        label = predict_image["label"]
        score = predict_image["score"]
        score_percent = score * 100

        # 결과를 세션 상태에 추가
        result_entry = {
            "날짜+시각": now,
            "이미지": img_thumbnail,
            "predict_image": f"{label} ({score_percent:.1f}%)",
            "label": label,
            "score": score,
            "사용모델": selected_model,
        }

        st.session_state.result_history.append(result_entry)

        # 현재 결과 표시 (상위 1개 강조)
        st.success("예측 완료!")
        st.markdown("### 예측 결과")

        # 상위 결과 강조 표시
        st.markdown(f"**예측된 클래스**: `{label}`")

        # 확률을 progress bar로 시각화
        st.progress(score)
        st.markdown(f"**신뢰도**: {score_percent:.1f}%")

    elif isinstance(predict_image, dict) and "text" in predict_image:
        # 텍스트 결과
        text_result = predict_image["text"]
        result_entry = {
            "날짜+시각": now,
            "이미지": img_thumbnail,
            "predict_image": text_result,
            "사용모델": selected_model,
        }

        st.session_state.result_history.append(result_entry)

        # 현재 결과 표시
        st.success("예측 완료!")
        st.write(f"**예측 결과**: {text_result}")
    else:
        # 기타 결과
        result_entry = {
            "날짜+시각": now,
            "이미지": img_thumbnail,
            "predict_image": str(predict_image),
            "사용모델": selected_model,
        }

        st.session_state.result_history.append(result_entry)

        # 현재 결과 표시
        st.success("예측 완료!")
        st.write(f"**예측 결과**: {str(predict_image)}")

    # 페이지 새로고침하여 결과 내역 테이블이 맨 위에 표시되도록
    st.rerun()


# 세션 상태로 결과 내역 관리
if "result_history" not in st.session_state:
    st.session_state.result_history = []

# 결과 내역 테이블을 맨 위에 표시
if st.session_state.result_history:
    st.markdown("---")
    st.markdown("#### 결과 내역")

    # 헤더
    header_cols = st.columns([2, 2, 2, 2])
    header_names = ["날짜+시각", "이미지", "예측 결과", "score"]
    for col, name in zip(header_cols, header_names):
        col.markdown(f"**{name}**")

    # 결과 내역 표시 (최신순)
    for row in reversed(st.session_state.result_history):
        cols = st.columns([2, 2, 2, 2])
        cols[0].write(row["날짜+시각"])
        cols[1].image(row["이미지"], width=48)

        # 예측 결과 표시 (label과 score가 있으면 강조)
        if "label" in row and "score" in row:
            # cols[2].markdown(f"**{row['label']}**\n\n{score_percent:.1f}%")
            cols[2].write(row["label"])
        else:
            cols[2].write(row["predict_image"])

        if "label" in row and "score" in row:
            score_percent = row["score"] * 100
            cols[3].write(f"{score_percent:.1f}%")
            cols[3].progress(round(score_percent))

    st.markdown("---")
