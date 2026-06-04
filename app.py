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

# Rembg Library (추가됨)
from rembg import remove, new_session

# SAM 2 Libraries
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =========================================================
# 1. Page Configuration & Session State
# =========================================================
st.set_page_config(
    page_title="Image to 3D Viewer",
    page_icon="🧊",
    layout="centered",
)

_ss = st.session_state

def _init(key, val):
    if key not in _ss:
        _ss[key] = val

_init("mode", None)            
_init("sam2_done", False)
_init("trellis_done", False) 
_init("extracted_image", None) 
_init("mesh_path", None)
_init("last_uploaded", None)

_init("modeb_keyframes", [])
_init("modeb_segmented_frames", [])
_init("modeb_sam2_done", False)
_init("modeb_lrm_done", False)

# =========================================================
# 2. AI Model Loaders (Cached Resource)
# =========================================================
@st.cache_resource
def load_sam2_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.normpath(os.path.join(base_dir, "checkpoints"))
    sam2_checkpoint = os.path.normpath(os.path.join(checkpoint_dir, "sam2.1_hiera_small.pt"))
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    if not os.path.exists(sam2_checkpoint):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        with st.spinner("SAM 2 모델 가중치 다운로드 중..."):
            urllib.request.urlretrieve(url, sam2_checkpoint)
            st.success("SAM 2 다운로드 완료!")
            time.sleep(1)
            
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(model), device

# =========================================================
# 3. Helper Functions 
# =========================================================
def get_inference_preset():
    vm = psutil.virtual_memory()
    avail = vm.available / (1024 ** 3)
    total = vm.total / (1024 ** 3)
    usable = max(0.5, avail - 1.5)
    
    if usable < 2.5:
        p = {"resolution": 128, "tier": "Minimal"}
    elif usable < 6:
        p = {"resolution": 192, "tier": "Low"}
    elif usable < 14:
        p = {"resolution": 256, "tier": "Medium"}
    else:
        p = {"resolution": 512, "tier": "High"}
        
    p["available_gb"] = avail
    p["total_gb"] = total
    return p

def extract_smart_keyframes(file_path, num_frames=6, target_size=(512, 512)):
    extracted_images = []
    if file_path.lower().endswith('.gif'):
        gif = Image.open(file_path)
        frames = []
        try:
            while True:
                frames.append(np.array(gif.convert("RGB")))
                gif.seek(len(frames))
        except EOFError:
            pass 
            
        total_frames = len(frames)
        if total_frames == 0: return []
            
        interval = max(1, total_frames // num_frames)
        num_frames = min(num_frames, total_frames)
        
        for i in range(num_frames):
            start_frame = i * interval
            end_frame = min((i + 1) * interval, total_frames)
            best_frame = None
            max_sharpness = -1
            step = max(1, (end_frame - start_frame) // 10)
            
            for f_idx in range(start_frame, end_frame, step):
                frame = frames[f_idx]
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    best_frame = frame
                    
            if best_frame is not None:
                img = Image.fromarray(best_frame)
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                extracted_images.append(img)
        return extracted_images

    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    
    for i in range(num_frames):
        start_frame = i * interval
        end_frame = min((i + 1) * interval, total_frames)
        best_frame = None
        max_sharpness = -1
        step = max(1, (end_frame - start_frame) // 10)
        
        for f_idx in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                best_frame = frame
                
        if best_frame is not None:
            best_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(best_frame)
            img.thumbnail(target_size, Image.Resampling.LANCZOS) 
            extracted_images.append(img)
            
    cap.release()
    return extracted_images

def _show_result(obj_path: str):
    if not obj_path or not os.path.exists(obj_path):
        st.error("생성된 .obj 메쉬 파일을 디스크에서 찾을 수 없습니다.")
        return

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"저장 경로: `{obj_path}`")
        with open(obj_path, "rb") as f:
            st.download_button(
                label="⬇️ .obj 메쉬 파일 다운로드",
                data=f.read(),
                file_name=os.path.basename(obj_path),
                mime="model/obj"
            )
    with col2:
        st.markdown(
            f'<a href="http://localhost:8502" target="_blank">'
            f'<button style="background:#22aa66;color:white;border:none;'
            f'padding:8px 20px;border-radius:8px;font-size:14px;cursor:pointer;width:100%;height:38px;">'
            f'🏠 다중 가상 쇼룸(Showroom) 열기 (새 탭)</button></a>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🖥️ 로컬 3D 뷰어 미리보기 (Three.js)")
    safe_path = urllib.parse.quote(obj_path.replace(os.sep, '/'))
    viewer_url = f"http://localhost:8502/?obj={safe_path}"
    components.iframe(viewer_url, height=650, scrolling=False)

# =========================================================
# 4. Streamlit User Interface
# =========================================================
with st.sidebar:
    st.subheader("🖥️ 시스템 자원 모니터링")
    _vm = psutil.virtual_memory()
    st.metric("총 RAM", f"{_vm.total / 1e9:.1f} GB")
    st.metric("가용 RAM", f"{_vm.available / 1e9:.1f} GB", delta=f"{_vm.percent}% 사용 중", delta_color="inverse")
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
    st.caption(f"🎮 하드웨어 가속 기기: `{gpu_info}`")
    _preset = get_inference_preset()
    st.caption(f"동적 할당 티어: **{_preset['tier']}**")
    st.divider()
    st.caption("독립 구동 터미널 가이드:")
    st.code("streamlit run showroom.py --server.port 8502")

st.title("🧊 Image to 3D Viewer")
st.write("2D 이미지 또는 영상을 업로드하면 AI가 3D 메쉬(.obj)로 변환합니다.")
st.divider()

st.subheader("🔀 변환 모드 선택")
col_a, col_b = st.columns(2)

with col_a:
    if st.button("🟦 Mode A\n단순 객체 (단일 사진)\nSAM 2 → TRELLIS"):
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _ss.mode = "A"
        _ss.sam2_done = False
        _ss.trellis_done = False
        _ss.extracted_image = None
        _ss.mesh_path = None

with col_b:
    if st.button("🟧 Mode B\n복잡한 가구 (비디오/GIF/다중 사진)\nSAM 2 Batch → LRM"):
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _ss.mode = "B"
        _ss.modeb_keyframes = []
        _ss.modeb_segmented_frames = []
        _ss.modeb_sam2_done = False
        _ss.modeb_lrm_done = False
        _ss.mesh_path = None

if _ss.mode is None:
    st.info("👆 분석하고자 하는 물체의 형태에 알맞은 변환 모드를 선택해주세요.")
    st.stop()

st.divider()

# =========================================================
# 5. Mode A Implementation (Hybrid Rembg + SAM 2)
# =========================================================
if _ss.mode == "A":
    st.markdown("### 🟦 Mode A — 단순 객체 단일 이미지 복원 (TRELLIS)")
    uploaded_file = st.file_uploader("단일 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"], key="modeA_upload")

    if uploaded_file:
        if _ss.last_uploaded != uploaded_file.name:
            _ss.sam2_done = False
            _ss.trellis_done = False
            _ss.extracted_image = None
            _ss.mesh_path = None
            _ss.last_uploaded = uploaded_file.name

        image = Image.open(uploaded_file)
        st.image(image, caption="업로드 원본 이미지", width="stretch")
        st.divider()

        st.subheader("Step 2: 배경 제거 및 객체 세그멘테이션 (Hybrid AI)")
        if st.button("SAM 2 실행", key="modeA_sam2") or _ss.sam2_done:
            if not _ss.sam2_done:
                with st.spinner("Rembg & SAM 2 기반 하이브리드 외곽 분석 중..."):
                    predictor, device = load_sam2_model()
                    img_rgb = np.array(image.convert("RGB"))
                    predictor.set_image(img_rgb)
                    h, w, _ = img_rgb.shape
                    
                    # [핵심 로직] 1. Rembg로 1차 실루엣(마스크) 추출
                    rembg_session = new_session("u2net")
                    rembg_mask = remove(img_rgb, session=rembg_session, only_mask=True)
                    
                    # [핵심 로직] 2. 실루엣에서 완벽한 타이트 바운딩 박스 계산
                    contours, _ = cv2.findContours(rembg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        x, y, bw, bh = cv2.boundingRect(c)
                        # 박스를 약간(10픽셀) 넉넉하게 주어 잘림 방지
                        box = np.array([[max(0, x-10), max(0, y-10), min(w, x+bw+10), min(h, y+bh+10)]])
                    else:
                        # 예외 처리: Rembg가 실패했을 경우 기존 중앙 박스 사용
                        box = np.array([[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]])

                    # [핵심 로직] 3. 추출된 완벽한 박스를 SAM 2에 전달하여 최종 픽셀 마스크 생성
                    masks, _, _ = predictor.predict(
                        point_coords=None, point_labels=None, box=box, multimask_output=False,
                    )
                    mask_2d = masks.squeeze()
                    alpha_channel = (mask_2d > 0).astype(np.uint8) * 255
                    
                    # OpenCV 후처리 (구멍 메우기 및 테두리 다듬기)
                    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        clean_mask = np.zeros_like(alpha_channel)
                        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                        alpha_channel = clean_mask

                    kernel = np.ones((5, 5), np.uint8)
                    alpha_channel = cv2.erode(alpha_channel, kernel, iterations=3)
                    alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)

                    img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    img_rgba[:, :, :3] = img_rgb
                    img_rgba[:, :, 3] = alpha_channel
                    
                    _ss.extracted_image = Image.fromarray(img_rgba, "RGBA")
                    _ss.sam2_done = True

            st.success("✅ 세그멘테이션 완료!")
            st.image(_ss.extracted_image, caption="노이즈 및 테두리가 제거된 객체 마스크", width="stretch")
            st.divider()

            st.subheader("Step 3: 3D 메쉬 생성 (TRELLIS)")
            if st.button("TRELLIS 모델 실행", key="modeA_trellis") or _ss.trellis_done:
                if not _ss.trellis_done:
                    from pipeline.trellis import run_modea_trellis
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
                    os.makedirs(output_dir, exist_ok=True)
                    preset = get_inference_preset()
                    
                    with st.spinner("🔄 TRELLIS: 단일 이미지 다각도 상상 및 3D 공간 복원 중..."):
                        raw_mesh_path = run_modea_trellis(_ss.extracted_image, output_dir, preset["resolution"])
                    
                    with st.spinner("✨ 3D 메쉬 표면 평탄화(Smoothing) 작업 중..."):
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
# 6. Mode B Implementation (Hybrid Rembg + Batch SAM 2)
# =========================================================
elif _ss.mode == "B":
    st.markdown("### 🟧 Mode B — 다각도 데이터 기반 정밀 복원 (InstantMesh LRM)")
    
    st.subheader("Step 1: 데이터 입력 (비디오, GIF 또는 다중 이미지)")
    tab1, tab2 = st.tabs(["🎥 동영상 및 GIF 업로드 (자동 추출)", "🖼️ 다중 이미지 직접 업로드"])
    
    with tab1:
        st.info("물체를 360도로 돌려가며 찍은 짧은 영상이나 움직이는 GIF를 업로드하세요.")
        uploaded_video = st.file_uploader("동영상 또는 GIF 파일 업로드", type=["mp4", "mov", "avi", "gif"])
        
        if uploaded_video:
            if st.button("핵심 프레임 추출하기", key="extract_video"):
                with st.spinner("미디어 분석 및 흔들림(Blur) 검사 중..."):
                    ext = os.path.splitext(uploaded_video.name)[1].lower()
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    tfile.write(uploaded_video.read())
                    tfile.flush()
                    
                    _ss.modeb_keyframes = extract_smart_keyframes(tfile.name, num_frames=6, target_size=(512, 512))
                    tfile.close()
                    os.unlink(tfile.name)
                    
                    _ss.modeb_sam2_done = False
                    _ss.modeb_lrm_done = False
                st.success("✅ 고품질 핵심 프레임 6장 추출 완료!")

    with tab2:
        st.info("객체의 앞, 뒤, 좌, 우 등을 찍은 사진을 4~6장 한꺼번에 업로드하세요.")
        uploaded_images = st.file_uploader("다중 이미지 업로드", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        if uploaded_images and len(uploaded_images) >= 2:
            if st.button("프레임 적용하기", key="apply_images"):
                with st.spinner("이미지 최적화 중..."):
                    imgs = []
                    for file in uploaded_images[:6]: 
                        img = Image.open(file).convert("RGB")
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        imgs.append(img)
                    _ss.modeb_keyframes = imgs
                    _ss.modeb_sam2_done = False
                    _ss.modeb_lrm_done = False
                st.success(f"✅ {len(imgs)}장의 핵심 프레임 등록 완료!")

    if _ss.modeb_keyframes:
        st.write("**[확보된 핵심 프레임]**")
        cols = st.columns(len(_ss.modeb_keyframes))
        for idx, img in enumerate(_ss.modeb_keyframes):
            cols[idx].image(img, width="stretch", caption=f"프레임 {idx+1}")
        
        st.divider()
        
        st.subheader("Step 2: 일괄 세그멘테이션 (Hybrid AI)")
        if st.button("모든 프레임 배경 제거", key="modeb_sam2") or _ss.modeb_sam2_done:
            if not _ss.modeb_sam2_done:
                predictor, device = load_sam2_model()
                
                # [성능 최적화] Rembg 세션은 반복문 밖에서 딱 한 번만 켭니다!
                rembg_session = new_session("u2net")
                
                processed_frames = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, frame_img in enumerate(_ss.modeb_keyframes):
                    status_text.text(f"프레임 {idx+1}/{len(_ss.modeb_keyframes)} 하이브리드 배경 제거 중...")
                    img_rgb = np.array(frame_img)
                    predictor.set_image(img_rgb)
                    h, w, _ = img_rgb.shape
                    
                    # 1. Rembg로 마스크 1초 컷 추출
                    rembg_mask = remove(img_rgb, session=rembg_session, only_mask=True)
                    
                    # 2. 박스 계산
                    contours, _ = cv2.findContours(rembg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        x, y, bw, bh = cv2.boundingRect(c)
                        # 박스를 여유 있게 주어 신발 코/뒷꿈치 잘림 완벽 방지
                        box = np.array([[max(0, x-10), max(0, y-10), min(w, x+bw+10), min(h, y+bh+10)]])
                    else:
                        box = np.array([[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]])
                    
                    # 3. 완벽한 박스로 SAM 2 구동
                    masks, _, _ = predictor.predict(
                        point_coords=None, point_labels=None, box=box, multimask_output=False,
                    )
                    mask_2d = masks.squeeze()
                    alpha_channel = (mask_2d > 0).astype(np.uint8) * 255
                    
                    # OpenCV 찌꺼기 제거
                    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        clean_mask = np.zeros_like(alpha_channel)
                        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                        alpha_channel = clean_mask
                        
                    kernel = np.ones((5, 5), np.uint8)
                    alpha_channel = cv2.erode(alpha_channel, kernel, iterations=3)
                    alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
                    
                    img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    img_rgba[:, :, :3] = img_rgb
                    img_rgba[:, :, 3] = alpha_channel
                    processed_frames.append(Image.fromarray(img_rgba, "RGBA"))
                    
                    progress_bar.progress((idx + 1) / len(_ss.modeb_keyframes))
                
                status_text.empty()
                _ss.modeb_segmented_frames = processed_frames
                _ss.modeb_sam2_done = True
                
                # 메모리 정리
                del predictor
                del rembg_session
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            st.success("✅ 모든 프레임 누끼 및 VRAM 정리 완료!")
            cols = st.columns(len(_ss.modeb_segmented_frames))
            for idx, img in enumerate(_ss.modeb_segmented_frames):
                cols[idx].image(img, width="stretch", caption=f"투명 프레임 {idx+1}")
            
            st.divider()
            
            st.subheader("Step 3: 다각도 3D 복원 (InstantMesh LRM)")
            if st.button("LRM 3D 엔진 구동", key="modeb_lrm") or _ss.modeb_lrm_done:
                if not _ss.modeb_lrm_done:
                    with st.spinner("🚀 InstantMesh 다각도 3D 복원 엔진 가동 중... (VRAM 집중 구간)"):
                        try:
                            from pipeline.instantmesh import run_modeb_instantmesh
                        except ModuleNotFoundError:
                            st.error("🚨 잠시만요! 아직 `pipeline/instantmesh.py` 파일을 만들지 않았습니다. 백엔드 AI 엔진 코드를 먼저 추가해야 합니다.")
                            st.stop()
                            
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
                        os.makedirs(output_dir, exist_ok=True)
                        
                        raw_mesh_path = run_modeb_instantmesh(_ss.modeb_segmented_frames, output_dir)
                        
                        with st.spinner("✨ 3D 메쉬 표면 정밀 평탄화 작업 중..."):
                            import open3d as o3d
                            mesh = o3d.io.read_triangle_mesh(raw_mesh_path)
                            mesh = mesh.filter_smooth_taubin(number_of_iterations=15)
                            mesh.compute_vertex_normals()
                            o3d.io.write_triangle_mesh(raw_mesh_path, mesh)
                            
                        _ss.mesh_path = raw_mesh_path
                        _ss.modeb_lrm_done = True

                st.success("✅ InstantMesh 다각도 3D 복원 완료!")
                _show_result(_ss.mesh_path)