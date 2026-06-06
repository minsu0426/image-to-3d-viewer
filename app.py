import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import time
import torch
import os
import urllib.request
import urllib.parse
import psutil
import gc
import cv2
import tempfile
import glob

# Rembg & SAM 2 Libraries
from rembg import remove, new_session
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =========================================================
# 1. Page Configuration & Session State
# =========================================================
st.set_page_config(page_title="3D Studio Pro", page_icon="🧊", layout="wide")

_ss = st.session_state

def _init(key, val):
    if key not in _ss: _ss[key] = val

_init("mode", None)            
_init("step", 1)
_init("mesh_path", None)

# Mode A States
_init("modea_image", None)
_init("sam2_done", False)
_init("trellis_done", False) 
_init("extracted_image", None) 

# Mode B States
_init("modeb_keyframes", [])
_init("modeb_segmented_frames", [])
_init("modeb_sam2_done", False)
_init("modeb_lrm_done", False)

# =========================================================
# 2. Helper Functions & AI Model
# =========================================================
@st.cache_resource
def load_sam2_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.normpath(os.path.join(base_dir, "checkpoints"))
    sam2_checkpoint = os.path.normpath(os.path.join(checkpoint_dir, "sam2.1_hiera_small.pt"))
    
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.exists(sam2_checkpoint):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        with st.spinner("SAM 2 모델 다운로드 중..."):
            urllib.request.urlretrieve(url, sam2_checkpoint)
            time.sleep(1)
            
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(model), device

def get_inference_preset():
    vm = psutil.virtual_memory()
    avail = vm.available / (1024 ** 3)
    if avail < 2.5: return {"resolution": 128, "tier": "Minimal"}
    elif avail < 6: return {"resolution": 192, "tier": "Low"}
    elif avail < 14: return {"resolution": 256, "tier": "Medium"}
    else: return {"resolution": 512, "tier": "High"}

def extract_smart_keyframes(file_path, num_frames=8, target_size=(512, 512)):
    extracted_images = []
    if file_path.lower().endswith('.gif'):
        gif = Image.open(file_path)
        frames = []
        try:
            while True:
                frames.append(np.array(gif.convert("RGB")))
                gif.seek(len(frames))
        except EOFError: pass 
        total_frames = len(frames)
        if total_frames == 0: return []
        interval = max(1, total_frames // num_frames)
        num_frames = min(num_frames, total_frames)
        for i in range(num_frames):
            start_frame, end_frame = i * interval, min((i + 1) * interval, total_frames)
            best_frame, max_sharpness = None, -1
            step = max(1, (end_frame - start_frame) // 10)
            for f_idx in range(start_frame, end_frame, step):
                frame = frames[f_idx]
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness > max_sharpness:
                    max_sharpness, best_frame = sharpness, frame
            if best_frame is not None:
                img = Image.fromarray(best_frame)
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                extracted_images.append(img)
        return extracted_images
    
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    for i in range(num_frames):
        start_frame, end_frame = i * interval, min((i + 1) * interval, total_frames)
        best_frame, max_sharpness = None, -1
        step = max(1, (end_frame - start_frame) // 10)
        for f_idx in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness > max_sharpness:
                max_sharpness, best_frame = sharpness, frame
        if best_frame is not None:
            best_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(best_frame)
            img.thumbnail(target_size, Image.Resampling.LANCZOS) 
            extracted_images.append(img)
    cap.release()
    return extracted_images

def _show_result(obj_path: str):
    if not obj_path or not os.path.exists(obj_path): return
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"저장 경로: `{obj_path}`")
        with open(obj_path, "rb") as f:
            st.download_button("⬇️ .obj 메쉬 파일 다운로드", data=f.read(), file_name=os.path.basename(obj_path), mime="model/obj")
    with col2:
        safe_path = urllib.parse.quote(obj_path.replace(os.sep, '/'))
        st.markdown(f'<a href="http://localhost:8502/?obj={safe_path}" target="_blank"><button style="background:#22aa66;color:white;border:none;padding:8px 20px;border-radius:8px;cursor:pointer;width:100%;height:42px;">🏠 단일 뷰어 / 다중 쇼룸 열기 (새 탭)</button></a>', unsafe_allow_html=True)

# =========================================================
# 3. Sidebar: Gallery & Status
# =========================================================
with st.sidebar:
    st.title("🧊 3D Studio Pro")
    st.caption("고품질 2D-to-3D 변환 파이프라인")
    
    st.subheader("🖥️ 시스템 상태")
    _vm = psutil.virtual_memory()
    st.progress(_vm.percent / 100.0, text=f"RAM 사용량 ({_vm.percent}%)")
    
    st.divider()
    st.subheader("📂 내 보관함 (Gallery)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    os.makedirs(mesh_dir, exist_ok=True)
    history_files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")), reverse=True)[:5]
    
    if history_files:
        for f in history_files:
            fname = os.path.basename(f)
            with st.expander(f"📦 {fname[:15]}..."):
                with open(f, "rb") as file_data:
                    st.download_button("⬇️ 다운로드", data=file_data, file_name=fname, key=f"dl_{fname}")
    else:
        st.info("아직 생성된 3D 모델이 없습니다.")
        
    st.divider()
    if st.button("🔄 처음으로 (모드 선택)"):
        _ss.mode = None
        _ss.step = 1
        st.rerun()

# =========================================================
# 4. Mode Selection & Wizard UI
# =========================================================
if _ss.mode is None:
    st.header("🔀 변환 모드 선택")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("단순한 형태의 객체나 정면 사진 1장만 있을 때 유리합니다.")
        if st.button("🟦 Mode A\n단일 이미지 복원 (TRELLIS)"):
            _ss.mode, _ss.step = "A", 1
            _ss.sam2_done, _ss.trellis_done = False, False
            _ss.modea_image, _ss.extracted_image = None, None
            st.rerun()
    with col_b:
        st.success("복잡한 가구나 비대칭 객체의 영상/다중 사진이 있을 때 완벽합니다.")
        if st.button("🟧 Mode B\n다각도 정밀 복원 (InstantMesh)"):
            _ss.mode, _ss.step = "B", 1
            _ss.modeb_keyframes, _ss.modeb_segmented_frames = [], []
            _ss.modeb_sam2_done, _ss.modeb_lrm_done = False, False
            st.rerun()
    st.stop()

# ----------------- 진행률 바 -----------------
st.progress(_ss.step / 3.0, text=f"Step {_ss.step} of 3")

# =========================================================
# Mode A: TRELLIS Logic
# =========================================================
if _ss.mode == "A":
    if _ss.step == 1:
        st.header("Step 1. 단일 이미지 업로드")
        uploaded_file = st.file_uploader("단일 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            _ss.modea_image = Image.open(uploaded_file)
            st.image(_ss.modea_image, caption="업로드 원본")
            if st.button("다음 단계로 이동 (배경 제거) ➔", type="primary"):
                _ss.sam2_done = False
                _ss.step = 2
                st.rerun()

    elif _ss.step == 2:
        st.header("Step 2. AI 배경 제거")
        if not _ss.sam2_done:
            with st.spinner("Rembg & SAM 2 기반 하이브리드 세그멘테이션 중..."):
                predictor, device = load_sam2_model()
                img_rgb = np.array(_ss.modea_image.convert("RGB"))
                predictor.set_image(img_rgb)
                h, w, _ = img_rgb.shape
                
                rembg_session = new_session("u2net")
                rembg_mask = remove(img_rgb, session=rembg_session, only_mask=True)
                contours, _ = cv2.findContours(rembg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(c)
                    box = np.array([[max(0, x-10), max(0, y-10), min(w, x+bw+10), min(h, y+bh+10)]])
                else: box = np.array([[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]])

                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                mask_2d = masks.squeeze()
                alpha_channel = (mask_2d > 0).astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    clean_mask = np.zeros_like(alpha_channel)
                    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    alpha_channel = clean_mask

                alpha_channel = cv2.erode(alpha_channel, np.ones((5, 5), np.uint8), iterations=3)
                alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)

                img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                img_rgba[:, :, :3], img_rgba[:, :, 3] = img_rgb, alpha_channel
                _ss.extracted_image = Image.fromarray(img_rgba, "RGBA")
                _ss.sam2_done = True
                del predictor, rembg_session
                gc.collect()

        # 🔥 기존 방식으로 복구: 캔버스 없애고 중앙에 깔끔하게 배치
        st.success("✅ 세그멘테이션 완료!")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(_ss.extracted_image, caption="노이즈 및 테두리가 제거된 객체 마스크")
            
        if st.button("다음 단계로 이동 (3D 생성) ➔", type="primary"):
            _ss.step = 3
            st.rerun()

    elif _ss.step == 3:
        st.header("Step 3. 3D 메쉬 생성 (TRELLIS)")
        if not _ss.trellis_done:
            with st.spinner("🔄 TRELLIS: 단일 이미지 다각도 상상 및 3D 공간 복원 중..."):
                from pipeline.trellis import run_modea_trellis
                base_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
                os.makedirs(output_dir, exist_ok=True)
                preset = get_inference_preset()
                raw_mesh_path = run_modea_trellis(_ss.extracted_image, output_dir, preset["resolution"])
                
                with st.spinner("✨ 3D 메쉬 표면 평탄화(Smoothing) 중..."):
                    import open3d as o3d
                    mesh = o3d.io.read_triangle_mesh(raw_mesh_path)
                    mesh = mesh.filter_smooth_taubin(number_of_iterations=20)
                    mesh.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(raw_mesh_path, mesh)
                    
                _ss.mesh_path = raw_mesh_path
                _ss.trellis_done = True
        st.success("✅ 3D 메쉬 생성이 정상 완료되었습니다!")
        _show_result(_ss.mesh_path)

# =========================================================
# Mode B: InstantMesh Logic
# =========================================================
elif _ss.mode == "B":
    if _ss.step == 1:
        st.header("Step 1. 다중 데이터 업로드 및 기준 프레임 설정")
        tab1, tab2 = st.tabs(["🎥 동영상/GIF 업로드", "🖼️ 다중 이미지 업로드"])
        
        with tab1:
            uploaded_video = st.file_uploader("동영상 또는 GIF 파일 업로드", type=["mp4", "mov", "avi", "gif"])
            if uploaded_video:
                num_extract = st.slider("📸 추출할 프레임 수", 6, 15, 8)
                if st.button("프레임 추출", type="primary"):
                    with st.spinner("프레임 추출 중..."):
                        ext = os.path.splitext(uploaded_video.name)[1].lower()
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                        tfile.write(uploaded_video.read())
                        tfile.flush()
                        _ss.modeb_keyframes = extract_smart_keyframes(tfile.name, num_frames=num_extract)
                        tfile.close()
                        os.unlink(tfile.name)
                        _ss.modeb_sam2_done = False
                    st.rerun()

        with tab2:
            uploaded_images = st.file_uploader("다중 이미지 업로드", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if uploaded_images and len(uploaded_images) >= 2:
                if st.button("프레임 적용", type="primary"):
                    imgs = []
                    for file in uploaded_images[:6]: 
                        img = Image.open(file).convert("RGB")
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        imgs.append(img)
                    _ss.modeb_keyframes = imgs
                    _ss.modeb_sam2_done = False
                    st.rerun()

        if _ss.modeb_keyframes:
            st.divider()
            cols = st.columns(len(_ss.modeb_keyframes))
            for idx, img in enumerate(_ss.modeb_keyframes):
                with cols[idx]:
                    st.markdown(f"<p style='text-align: center; color: {'#ff4b4b' if idx==0 else 'gray'}; font-weight: bold;'>{'⭐️ 메인 뷰' if idx==0 else '서브'}</p>", unsafe_allow_html=True)
                    st.image(img)
                    b1, b2 = st.columns(2)
                    if b1.button("◀", key=f"l_{idx}", disabled=(idx == 0)):
                        _ss.modeb_keyframes[idx], _ss.modeb_keyframes[idx-1] = _ss.modeb_keyframes[idx-1], _ss.modeb_keyframes[idx]
                        _ss.modeb_sam2_done = False
                        st.rerun()
                    if b2.button("▶", key=f"r_{idx}", disabled=(idx == len(_ss.modeb_keyframes)-1)):
                        _ss.modeb_keyframes[idx], _ss.modeb_keyframes[idx+1] = _ss.modeb_keyframes[idx+1], _ss.modeb_keyframes[idx]
                        _ss.modeb_sam2_done = False
                        st.rerun()
            st.divider()
            if st.button("다음 단계로 이동 (배경 제거) ➔", type="primary"):
                _ss.step = 2
                st.rerun()

    elif _ss.step == 2:
        st.header("Step 2. 메인 뷰 배경 제거 (초고속)")
        if not _ss.modeb_sam2_done:
            with st.spinner("메인 프레임 1장 집중 세그멘테이션 중..."):
                predictor, device = load_sam2_model()
                rembg_session = new_session("u2net")
                main_frame = _ss.modeb_keyframes[0]
                img_rgb = np.array(main_frame)
                predictor.set_image(img_rgb)
                h, w, _ = img_rgb.shape
                
                rembg_mask = remove(img_rgb, session=rembg_session, only_mask=True)
                contours, _ = cv2.findContours(rembg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(c)
                    box = np.array([[max(0, x-10), max(0, y-10), min(w, x+bw+10), min(h, y+bh+10)]])
                else: box = np.array([[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]])
                
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                mask_2d = masks.squeeze()
                alpha_channel = (mask_2d > 0).astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    clean_mask = np.zeros_like(alpha_channel)
                    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    alpha_channel = clean_mask
                    
                alpha_channel = cv2.erode(alpha_channel, np.ones((5, 5), np.uint8), iterations=3)
                alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
                
                img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                img_rgba[:, :, :3], img_rgba[:, :, 3] = img_rgb, alpha_channel
                
                _ss.modeb_segmented_frames = [Image.fromarray(img_rgba, "RGBA")]
                _ss.modeb_sam2_done = True
                del predictor, rembg_session
                gc.collect()

        # 🔥 기존 방식으로 복구: 캔버스 없애고 중앙에 깔끔하게 배치
        st.success("✅ 메인 뷰 누끼 추출 완료!")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(_ss.modeb_segmented_frames[0], caption="3D 복원 기준이 될 투명 메인 뷰")
            
        if st.button("다음 단계로 이동 (3D 생성) ➔", type="primary"):
            _ss.step = 3
            st.rerun()

    elif _ss.step == 3:
        st.header("Step 3. 다각도 3D 복원 (InstantMesh LRM)")
        if not _ss.modeb_lrm_done:
            with st.spinner("🚀 InstantMesh 다각도 3D 복원 엔진 가동 중..."):
                from pipeline.instantmesh import run_modeb_instantmesh
                base_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
                raw_mesh_path = run_modeb_instantmesh(_ss.modeb_segmented_frames, output_dir)
                
                with st.spinner("✨ 3D 메쉬 표면 정밀 평탄화 작업 중..."):
                    import open3d as o3d
                    mesh = o3d.io.read_triangle_mesh(raw_mesh_path)
                    mesh = mesh.filter_smooth_taubin(number_of_iterations=3)
                    mesh.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(raw_mesh_path, mesh)
                    
                _ss.mesh_path = raw_mesh_path
                _ss.modeb_lrm_done = True
                
        st.success("✅ 3D 메쉬 생성 완료!")
        _show_result(_ss.mesh_path)