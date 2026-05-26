import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import time
import torch
import os
import urllib.request
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

_init("mode", None)            # "A" (TripoSR) or "B" (InstantMesh)
_init("sam2_done", False)
_init("triposr_done", False)
_init("modeb_done", False)
_init("extracted_image", None) # Extracted PIL RGBA Image
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


@st.cache_resource
def load_triposr_model():
    from tsr.system import TSR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.normpath(os.path.join(base_dir, "checkpoints", "triposr"))
    
    if not os.path.exists(ckpt_dir):
        with st.spinner("TripoSR 모델 다운로드 중... (약 2GB)"):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="stabilityai/TripoSR",
                local_dir=ckpt_dir,
                ignore_patterns=["*.md", "*.txt"],
            )
            
    model = TSR.from_pretrained(ckpt_dir, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(131072)
    model.to(device)
    return model, device

# =========================================================
# 3. Dynamic Environment Preset (RAM-based Optimization)
# =========================================================

def get_inference_preset():
    vm = psutil.virtual_memory()
    avail = vm.available / (1024 ** 3)
    total = vm.total / (1024 ** 3)
    usable = max(0.5, avail - 1.5)
    
    if usable < 2.5:
        p = {"resolution": 32,  "chunk_size": 1024,   "tier": "Minimal"}
    elif usable < 6:
        p = {"resolution": 64,  "chunk_size": 4096,   "tier": "Low"}
    elif usable < 14:
        p = {"resolution": 128, "chunk_size": 16384,  "tier": "Medium"}
    elif usable < 30:
        p = {"resolution": 192, "chunk_size": 65536,  "tier": "High"}
    else:
        p = {"resolution": 256, "chunk_size": 131072, "tier": "Ultra"}
        
    p["available_gb"] = avail
    p["total_gb"] = total
    return p

# =========================================================
# 4. Core Inference Pipelines (Mode A & Mode B)
# =========================================================

def run_modea_inference(input_data) -> str:
    """ Mode A: Single Image input -> Patched TripoSR Core Model Pipeline """
    model, device = load_triposr_model()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    os.makedirs(output_dir, exist_ok=True)

    preset = get_inference_preset()
    st.info(f"🖥️ 프리셋 적용: **{preset['tier']}** (resolution: {preset['resolution']})")

    # Input Data Type Normalization (Numpy array or PIL Image -> PIL Image)
    if isinstance(input_data, Image.Image):
        pil_image = input_data
    else:
        pil_image = Image.fromarray(input_data, "RGBA")
        
    pil_image = pil_image.resize((512, 512), Image.LANCZOS)
    bg = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
    bg.paste(pil_image, mask=pil_image.split()[3])
    input_image = bg.convert("RGB")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        model.renderer.set_chunk_size(preset["chunk_size"])
        scene_codes = model([input_image], device=device)

    del input_image, bg, pil_image
    gc.collect()

    model.renderer.set_chunk_size(max(512, preset["chunk_size"] // 4))
    use_vc = preset["tier"] not in ("Minimal", "Low")

    try:
        # Python 3.13 호환 커스텀 패치: has_vertex_color 명시적 주입 및 resolution 조절
        meshes = model.extract_mesh(scene_codes, has_vertex_color=use_vc, resolution=preset["resolution"])
    except (MemoryError, RuntimeError) as e:
        st.warning(f"⚠️ 메모리 부족으로 인한 자동 폴백 연산 진행... ({e})")
        gc.collect()
        fallback_res = max(32, preset["resolution"] // 2)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=fallback_res)

    mesh = meshes[0]
    del scene_codes, meshes
    gc.collect()

    timestamp = int(time.time())
    obj_path = os.path.normpath(os.path.join(output_dir, f"mesh_modeA_{timestamp}.obj"))
    mesh.export(obj_path)
    return obj_path


def run_modeb_inference(images: list, video_path: str = None) -> str:
    """ Mode B: Multi-view images or Video -> InstantMesh Pipeline """
    try:
        from pipeline.instantmesh import run_modeb_pipeline
    except ImportError:
        st.error("🚨 pipeline/instantmesh.py 모듈을 찾을 수 없습니다. 파이프라인 구현 파일을 확인하세요.")
        st.stop()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    os.makedirs(output_dir, exist_ok=True)

    preset = get_inference_preset()
    resolution = preset["resolution"]

    if video_path:
        obj_path = run_modeb_pipeline(input_data=video_path, output_dir=output_dir, resolution=resolution)
    else:
        obj_path = run_modeb_pipeline(
            input_data=images, 
            output_dir=output_dir, 
            resolution=resolution,
            use_zero123_for_single=(len(images) == 1)
        )
    return obj_path

# =========================================================
# 5. Streamlit User Interface
# =========================================================

# --- Sidebar Component ---
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
    st.code("streamlit run viewer.py --server.port 8502")
    st.code("streamlit run showroom.py --server.port 8503")

# --- Main Header ---
st.title("🧊 Image to 3D Viewer")
st.write("2D 이미지 또는 영상을 업로드하면 AI가 3D 메쉬(.obj)로 변환합니다.")
st.divider()

# --- Mode Selector (with VRAM/RAM Memory Flush Logic) ---
st.subheader("🔀 변환 모드 선택")
col_a, col_b = st.columns(2)

with col_a:
    if st.button("🟦 Mode A\n단순 객체\n(캔, 큐브 등)\n단일 이미지 → TripoSR", use_container_width=True):
        # 모드 스위칭 시 캐시 삭제 및 메모리 가비지 컬렉션으로 OOM 완전 방어
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        _ss.mode = "A"
        _ss.sam2_done = False
        _ss.triposr_done = False
        _ss.modeb_done = False
        _ss.extracted_image = None
        _ss.mesh_path = None

with col_b:
    if st.button("🟧 Mode B\n복잡한 가구\n(침대, 책상 등)\n다중 이미지/영상 → InstantMesh", use_container_width=True):
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        _ss.mode = "B"
        _ss.sam2_done = False
        _ss.triposr_done = False
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
# 6. Mode A Implementation (SAM 2 + TripoSR)
# =========================================================

if _ss.mode == "A":
    st.markdown("### 🟦 Mode A — 단순 객체 (TripoSR)")
    
    st.subheader("Step 1: 이미지 업로드")
    uploaded_file = st.file_uploader("단일 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"], key="modeA_upload")

    if uploaded_file:
        if _ss.last_uploaded != uploaded_file.name:
            _ss.sam2_done = False
            _ss.triposr_done = False
            _ss.extracted_image = None
            _ss.mesh_path = None
            _ss.last_uploaded = uploaded_file.name

        image = Image.open(uploaded_file)
        st.image(image, caption="업로드 원본 이미지", use_column_width=True)
        st.divider()

        # Step 2: SAM2 (Box Prompt Patch)
        st.subheader("Step 2: 배경 제거 및 객체 세그멘테이션 (SAM 2)")
        if st.button("SAM 2 실행", key="modeA_sam2") or _ss.sam2_done:
            if not _ss.sam2_done:
                with st.spinner("SAM 2 기반 객체 외곽 분석 및 배경 분리 중..."):
                    predictor, device = load_sam2_model()
                    img_rgb = np.array(image.convert("RGB"))
                    predictor.set_image(img_rgb)
                    h, w, _ = img_rgb.shape
                    
                    # 텍스트 파편화 방지를 위한 가상의 중앙 Box 가이드라인 맵핑
                    box = np.array([[int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)]])
                    masks, _, _ = predictor.predict(
                        point_coords=None, point_labels=None,
                        box=box, multimask_output=False,
                    )
                    mask_2d = masks.squeeze()
                    img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    img_rgba[:, :, :3] = img_rgb
                    img_rgba[:, :, 3] = (mask_2d > 0).astype(np.uint8) * 255
                    _ss.extracted_image = Image.fromarray(img_rgba, "RGBA")
                    _ss.sam2_done = True

            st.success("✅ 세그멘테이션 완료!")
            st.image(_ss.extracted_image, caption="알파 채널 마스크가 적용된 객체", use_column_width=True)
            st.divider()

            # Step 3: TripoSR Inference
            st.subheader("Step 3: 3D 메쉬 생성 및 추론 (TripoSR)")
            if st.button("TripoSR 실행", key="modeA_triposr") or _ss.triposr_done:
                if not _ss.triposr_done:
                    with st.spinner("TripoSR 가중치 엔진 로드 중..."):
                        load_triposr_model()
                    with st.spinner("🔄 2D 고속 3D 공간 복원 연산 가동 중..."):
                        mesh_path = run_modea_inference(_ss.extracted_image)
                    _ss.mesh_path = mesh_path
                    _ss.triposr_done = True

                st.success("✅ 3D 메쉬 생성이 정상 완료되었습니다!")
                _show_result(_ss.mesh_path)

# =========================================================
# 7. Mode B Implementation (Multi-view / Video + InstantMesh)
# =========================================================

elif _ss.mode == "B":
    st.markdown("### 🟧 Mode B — 복잡한 가구 (InstantMesh)")
    
    input_type = st.radio(
        "입력 소스 형태 선택",
        ["📷 다중 이미지 (2~8장)", "🎥 동영상 (.mp4)"],
        key="modeb_input_type", horizontal=True,
    )
    st.divider()

    if input_type == "📷 다중 이미지 (2~8장)":
        st.subheader("Step 1: 다각도 소스 이미지 업로드")
        uploaded_files = st.file_uploader(
            "동일 객체의 다각도 촬영 사진들을 동시 선택하여 업로드하세요.",
            type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="modeb_images_upload",
        )

        if uploaded_files:
            images = [Image.open(f) for f in uploaded_files]
            _ss.modeb_images = images
            _ss.video_path = None

            cols = st.columns(min(len(images), 4))
            for i, img in enumerate(images):
                cols[i % 4].image(img, caption=f"Angle {i+1}", use_column_width=True)

            st.success(f"✅ 총 {len(images)}장의 소스 시퀀스 업로드 완료")
            st.divider()

            st.subheader("Step 2: 3D 메쉬 생성 (InstantMesh)")
            if st.button("InstantMesh 실행", key="modeb_run") or _ss.modeb_done:
                if not _ss.modeb_done:
                    with st.spinner("🔄 다각도 피처 매칭 기반 입체 복원 진행 중... (1~3분 소요)"):
                        mesh_path = run_modeb_inference(images=_ss.modeb_images)
                    _ss.mesh_path = mesh_path
                    _ss.modeb_done = True

                st.success("✅ 3D 가구 메쉬 복원 완료!")
                _show_result(_ss.mesh_path)

    else: # Video Framework Pipeline
        st.subheader("Step 1: 360도 촬영 동영상 업로드")
        video_file = st.file_uploader("동영상을 선택하세요.", type=["mp4", "avi", "mov"], key="modeb_video_upload")

        if video_file:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            video_dir = os.path.normpath(os.path.join(base_dir, "outputs", "videos"))
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.normpath(os.path.join(video_dir, video_file.name))

            with open(video_path, "wb") as f:
                f.write(video_file.read())

            _ss.video_path = video_path
            _ss.modeb_images = []

            st.video(video_file)
            st.success(f"✅ 비디오 버퍼 저장 완료: `{video_file.name}`")
            st.divider()

            st.subheader("Step 2: 프레임 추출 및 3D 메쉬 생성 (InstantMesh)")
            if st.button("InstantMesh 실행", key="modeb_video_run") or _ss.modeb_done:
                if not _ss.modeb_done:
                    with st.spinner("🎬 비디오 키프레임 샘플링 및 데이터 정형화 중..."):
                        pass
                    with st.spinner("🔄 시퀀스 3D 통합 볼륨 복원 연산 수행 중..."):
                        mesh_path = run_modeb_inference(images=[], video_path=_ss.video_path)
                    _ss.mesh_path = mesh_path
                    _ss.modeb_done = True

                st.success("✅ 영상 기반 3D 공간 복원 완료!")
                _show_result(_ss.mesh_path)

# =========================================================
# 8. Integrated Output & Multi-Port Embed Viewer
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
        # 가상 쇼룸은 파라미터 없이 순수 8502 포트로 새 창을 열어줍니다.
        st.markdown(
            f'<a href="http://localhost:8502" target="_blank">'
            f'<button style="background:#22aa66;color:white;border:none;'
            f'padding:8px 20px;border-radius:8px;font-size:14px;cursor:pointer;width:100%;height:38px;">'
            f'🏠 다중 가상 쇼룸(Showroom) 열기 (새 탭)</button></a>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 💡 통합된 8502 포트 뷰어의 ?obj= 파라미터를 사용해 인라인 미리보기를 출력합니다.
    st.subheader("🖥️ 로컬 3D 뷰어 미리보기 (Three.js)")
    viewer_url = f"http://localhost:8502/?obj={obj_path.replace(os.sep, '/')}"
    
    components.iframe(viewer_url, height=650, scrolling=False)
    st.caption("※ 뷰어가 회색 화면으로 보이면 터미널에서 `streamlit run showroom.py --server.port 8502`가 구동 중인지 확인하세요.")