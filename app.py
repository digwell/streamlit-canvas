from streamlit_drawable_canvas import st_canvas

import streamlit as st
from src.my_model import MyModel


@st.cache_resource
def get_model_instance():
    return MyModel()

st.set_page_config(page_title="미션15", page_icon="🚀")

st.title("필기 인식")
st.write("0~9 숫자 중 하나를 손으로 써주세요.")

with st.form("form1"):
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=250,
        width=250,
        drawing_mode="freedraw",
        key="canvas",
    )

    import socket
    ip_address = socket.gethostbyname(socket.gethostname())
    st.write(f"현재 시스템의 IP 주소: {ip_address}")

    ai_type = st.radio("분석기 구분", ["digit", "alphabet", "hangul"])

    models = []
    if ip_address == "127.0.0.1":
        models.append("ollama - gemma3:12b")
        models.append("ollama - llama3.2-vision:11b")

    models.append("huggingface - google/vit-base-patch16-224")
    # models.append("huggingface - ddobokki/ko-trocr")
    # models.append("huggingface - LGAI-EXAONE/K-EXAONE-236B-A23B")
    selected_model = st.radio("모델 선택", models)

    submit_button = st.form_submit_button("Submit")



if submit_button:
    my_model = get_model_instance()
    image_data = canvas_result.image_data

    predict_image = my_model.predict_image(ai_type, selected_model, image_data)
    st.write(predict_image)
