import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# --- 1. 페이지 및 Session State (기억 장치) 초기화 ---
st.set_page_config(page_title="Image to 3D Viewer", page_icon="🧊", layout="centered")

# 버튼을 눌렀던 상태를 기억하는 변수 생성
if 'sam2_done' not in st.session_state:
    st.session_state.sam2_done = False
if 'triposr_done' not in st.session_state:
    st.session_state.triposr_done = False

st.title("🧊 Image to 3D Viewer")
st.write("2D 이미지를 업로드하면 SAM 2로 객체를 추출하고 TripoSR을 통해 3D 모델로 변환합니다.")

# --- 2. 이미지 업로드 섹션 ---
st.subheader("Step 1: 이미지 업로드")
uploaded_file = st.file_uploader("2D 이미지 파일(jpg, png)을 선택하세요.", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지를 새로 올리면 이전 작업 상태를 초기화
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.sam2_done = False
        st.session_state.triposr_done = False
        st.session_state.last_uploaded = uploaded_file.name

    # 원본 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 원본 이미지", use_column_width=True)
    st.divider()
    
    # --- 3. 객체 추출 (SAM 2) 영역 ---
    st.subheader("Step 2: 배경 제거 및 객체 추출 (SAM 2)")
    
    # 버튼을 누르거나, 이미 완료된 상태라면 실행
    if st.button("SAM 2 실행 (객체 추출)") or st.session_state.sam2_done:
        
        # 아직 처리가 안 된 상태일 때만 로딩 애니메이션 보여주기
        if not st.session_state.sam2_done:
            with st.spinner("SAM 2 모델이 객체를 분석하고 있습니다..."):
                time.sleep(2) # 실제 모델 연산 시뮬레이션
                st.session_state.sam2_done = True # 작업 완료 상태 저장!

        st.success("객체 추출 완료!")
        st.info("이곳에 배경이 투명해진(RGBA) 객체 이미지가 표시됩니다.")
        st.divider()
            
        # --- 4. 3D 변환 (TripoSR) 영역 ---
        # (Step 2가 완료된 상태에서만 화면에 표시됨)
        st.subheader("Step 3: 3D 메쉬 생성 및 뷰어 (TripoSR)")
        
        if st.button("TripoSR 실행 (3D 변환)") or st.session_state.triposr_done:
            
            if not st.session_state.triposr_done:
                with st.spinner("TripoSR 모델이 3D 메쉬를 생성하고 있습니다..."):
                    time.sleep(3)
                    st.session_state.triposr_done = True # 작업 완료 상태 저장!
                    
            st.success("3D 변환 완료!")
            st.info("이곳에 변환된 3D 모델(.obj)을 돌려볼 수 있는 인터랙티브 뷰어가 나타납니다.")