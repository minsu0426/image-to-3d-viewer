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
_init("modeb_done", False)
_init("extracted_image", None) 
_init("mesh_path", None)
_init("last_uploaded", None)
_init("modeb_images", [])      
_init("video_path", None)      

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
# 3. Dynamic Environment Preset
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

# =========================================================
# 4. Helper Functions
# =========================================================
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
                mime="model/obj",
                key=f"dl_obj_{os.path.basename(obj_path)}",
                use_container_width=True
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
    st.caption("※ 뷰어가 회색 화면으로 보이면 터미널에서 `streamlit run showroom.py --server.port 8502`가 구동 중인지 확인하세요.")

# =========================================================
# 5. Streamlit User Interface
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
    if st.button("🟦 Mode A\n단순 객체 (단일 사진)\nSAM 2 → TRELLIS", use_container_width=True):
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        _ss.mode = "A"
        _ss.sam2_done = False
        _ss.trellis_done = False
        _ss.modeb_done = False
        _ss.extracted_image = None
        _ss.mesh_path = None

with col_b:
    if st.button("🟧 Mode B\n복잡한 가구 (다중 사진/영상)\nInstantMesh", use_container_width=True):
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        _ss.mode = "B"
        _ss.sam2_done = False
        _ss.trellis_done = False
        _ss.modeb_done = False
        _ss.extracted_image = None
        _ss.mesh_path = None
        _ss.modeb_images = []
        _ss.video_path = None

if _ss.mode is None:
    st.info("👆 분석하고자 하는 물체의 형태에 알맞은 변환 모드를 선택해주세요.")
    st.stop()

st.divider()

# =========================================================
# 6. Mode A Implementation (SAM 2 + TRELLIS)
# =========================================================
if _ss.mode == "A":
    st.markdown("### 🟦 Mode A — 단순 객체 다각도 상상 복원 (TRELLIS)")
    
    st.subheader("Step 1: 이미지 업로드")
    uploaded_file = st.file_uploader("단일 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"], key="modeA_upload")

    if uploaded_file:
        if _ss.last_uploaded != uploaded_file.name:
            _ss.sam2_done = False
            _ss.trellis_done = False
            _ss.extracted_image = None
            _ss.mesh_path = None
            _ss.last_uploaded = uploaded_file.name

        image = Image.open(uploaded_file)
        st.image(image, caption="업로드 원본 이미지", use_column_width=True)
        st.divider()

        st.subheader("Step 2: 배경 제거 및 객체 세그멘테이션 (SAM 2)")
        if st.button("SAM 2 실행", key="modeA_sam2") or _ss.sam2_done:
            if not _ss.sam2_done:
                with st.spinner("SAM 2 기반 객체 외곽 분석 및 배경 분리 중..."):
                    import cv2
                    predictor, device = load_sam2_model()
                    img_rgb = np.array(image.convert("RGB"))
                    predictor.set_image(img_rgb)
                    h, w, _ = img_rgb.shape
                    
                    box = np.array([[int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)]])
                    masks, _, _ = predictor.predict(
                        point_coords=None, point_labels=None,
                        box=box, multimask_output=False,
                    )
                    mask_2d = masks.squeeze()
                    
                    # 1차 평탄화: 마스크 테두리 블러 처리
                    alpha_channel = (mask_2d > 0).astype(np.uint8) * 255
                    alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
                    
                    img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    img_rgba[:, :, :3] = img_rgb
                    img_rgba[:, :, 3] = alpha_channel
                    
                    _ss.extracted_image = Image.fromarray(img_rgba, "RGBA")
                    _ss.sam2_done = True

            st.success("✅ 세그멘테이션 완료!")
            st.image(_ss.extracted_image, caption="외곽선이 다듬어진 객체 마스크", use_column_width=True)
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
                    
                    # 💡 2차 평탄화: Open3D를 이용한 3D 메쉬 표면 다림질 작업
                    with st.spinner("✨ 3D 메쉬 표면 평탄화(Smoothing) 작업 중..."):
                        import open3d as o3d
                        mesh = o3d.io.read_triangle_mesh(raw_mesh_path)
                        # Taubin 스무딩: 부피 수축을 방지하면서 뾰족하고 거친 표면만 펴주는 고급 알고리즘
                        mesh = mesh.filter_smooth_taubin(number_of_iterations=20)
                        # 빛 반사가 자연스럽도록 노말(법선) 재계산
                        mesh.compute_vertex_normals()
                        # 다림질 완료된 모델 덮어쓰기
                        o3d.io.write_triangle_mesh(raw_mesh_path, mesh)
                        
                    _ss.mesh_path = raw_mesh_path
                    _ss.trellis_done = True

                st.success("✅ 3D 메쉬 평탄화 및 생성이 정상 완료되었습니다!")
                _show_result(_ss.mesh_path)

# =========================================================
# 7. Mode B Implementation (InstantMesh - Placeholder)
# =========================================================
elif _ss.mode == "B":
    st.markdown("### 🟧 Mode B — 복잡한 가구 (InstantMesh)")
    st.info("🚧 파이프라인 개발 진행 중입니다...")