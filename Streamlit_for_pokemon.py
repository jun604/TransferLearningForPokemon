import streamlit as st
import torch
import cv2 as cv
import numpy as np
import os
import torch_directml

# 기존 파일(transfer_learning_for_pokemon)에서 함수와 설정을 가져옵니다.
from transfer_learning_for_pokemon import GetModel, WhatIsThisPokemon, NameMaping

# --- 환경 설정 ---
if torch_directml.is_available():
    device = torch_directml.device()
else:
    device = torch.device('cpu')

st.set_page_config(page_title="포켓몬 분류 데모", layout="wide")

# --- 리소스 로드 (캐싱) ---
@st.cache_resource
def load_essentials():
    # 데이터셋 구조에서 클래스 이름 추출
    from torchvision import datasets
    full_dataset = datasets.ImageFolder('dataset/PokemonData')
    categories = full_dataset.classes
    name_map = NameMaping()
    return categories, name_map

@st.cache_resource
def load_trained_model(weight_path):
    import torchvision.models as models
    # 파일명에 따라 모델 구조 결정
    if "resnet34" in weight_path:
        m = GetModel(models.resnet34, models.ResNet34_Weights.DEFAULT)
    else:
        m = GetModel(models.resnet18, models.ResNet18_Weights.DEFAULT)
    
    # 가중치 로드
    m.model.load_state_dict(torch.load(weight_path, map_location=device))
    m.model.eval()
    return m

# --- UI 구성 ---
categories, name_map = load_essentials()

st.title("🐾 Pokemon Classifier Demo")
st.markdown(f"현재 구동 장치: `{device}`")

# 사이드바: 모델 선택
model_files = {
    "Model 1 (ResNet34 Full)": "pokemon_resnet34_full.pth",
    "Model 2 (ResNet34 Frozen)": "pokemon_resnet34_finetuned.pth",
    "Model 3 (ResNet18 Full)": "pokemon_resnet18_full.pth",
    "Model 4 (ResNet18 Frozen)": "pokemon_resnet18_finetuned.pth"
}
selected_name = st.sidebar.selectbox("테스트할 모델을 선택하세요", list(model_files.keys()))
target_weight = model_files[selected_name]

if not os.path.exists(target_weight):
    st.error(f"!!! 모델 파일({target_weight})이 없습니다. 먼저 학습을 완료해 주세요.")
else:
    # 모델 로드
    ModelObj = load_trained_model(target_weight)
    
    # 이미지 업로드
    uploaded_file = st.file_uploader("포켓몬 사진을 업로드하세요...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # 파일 저장 및 예측
        with open("temp_input.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 내 파일의 WhatIsThisPokemon 함수 호출
        results = WhatIsThisPokemon(ModelObj, "temp_input.jpg", top_k=5)
        
        # 결과 화면 분할
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption='업로드된 이미지', use_column_width=True)
            
        with col2:
            st.subheader("🔍 분류 결과 (Top-5)")
            for prob, index in results:
                eng_name = categories[index]
                kor_name = name_map.get(eng_name, eng_name)
                
                st.write(f"**{kor_name}** ({eng_name})")
                st.progress(prob)
                st.write(f"확률: {prob*100:.2f}%")
                st.divider()

# 성능 지표 확인용 (사이드바)
st.sidebar.divider()
if st.sidebar.checkbox("학습 결과 그래프 확인"):
    if os.path.exists('learning_curve.png'):
        st.sidebar.image('learning_curve.png', caption='Loss Curve')
    if os.path.exists('performance_comparison.png'):
        st.sidebar.image('performance_comparison.png', caption='Performance')