import streamlit as st
import numpy as np
from PIL import Image
import time
import torch
import os
import urllib.request
import psutil
import gc

# SAM 2 라이브러리
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. 페이지 및 Session State 초기화 ---
st.set_page_config(page_title="Image to 3D Viewer", page_icon="🧊", layout="centered")

if 'sam2_done' not in st.session_state:
    st.session_state.sam2_done = False
if 'triposr_done' not in st.session_state:
    st.session_state.triposr_done = False
if 'extracted_image' not in st.session_state:
    st.session_state.extracted_image = None
if 'mesh_path' not in st.session_state:
    st.session_state.mesh_path = None


# =========================================================
# 모델 로딩 함수 (캐싱 적용)
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
        with st.spinner("SAM 2 모델 가중치를 다운로드하는 중입니다... (최초 1회)"):
            urllib.request.urlretrieve(url, sam2_checkpoint)
            st.success("모델 다운로드 완료!")
            time.sleep(1)
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor, device


@st.cache_resource
def load_triposr_model():
    from tsr.system import TSR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.normpath(os.path.join(base_dir, "checkpoints"))
    triposr_checkpoint = os.path.normpath(os.path.join(checkpoint_dir, "triposr"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(triposr_checkpoint):
        with st.spinner("TripoSR 모델 가중치를 다운로드하는 중입니다... (최초 1회, 약 2GB)"):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="stabilityai/TripoSR",
                local_dir=triposr_checkpoint,
                ignore_patterns=["*.md", "*.txt"],
            )
            st.success("TripoSR 모델 다운로드 완료!")
            time.sleep(1)
    model = TSR.from_pretrained(
        triposr_checkpoint,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(131072)
    model.to(device)
    return model, device


# =========================================================
# RAM 기반 자동 프리셋 선택
# =========================================================

def get_inference_preset():
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)
    usable_gb = max(0.5, available_gb - 1.5)
    if usable_gb < 2.5:
        preset = {"resolution": 32,  "chunk_size": 1024,   "tier": "Minimal"}
    elif usable_gb < 6:
        preset = {"resolution": 64,  "chunk_size": 4096,   "tier": "Low"}
    elif usable_gb < 14:
        preset = {"resolution": 128, "chunk_size": 16384,  "tier": "Medium"}
    elif usable_gb < 30:
        preset = {"resolution": 192, "chunk_size": 65536,  "tier": "High"}
    else:
        preset = {"resolution": 256, "chunk_size": 131072, "tier": "Ultra"}
    preset["available_gb"] = available_gb
    preset["total_gb"] = total_gb
    return preset


# =========================================================
# 3D 변환 실행 함수
# =========================================================

def run_triposr_inference(input_data) -> str:
    model, device = load_triposr_model()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    os.makedirs(output_dir, exist_ok=True)

    preset = get_inference_preset()
    st.info(
        f"🖥️ **자동 환경 분석**\n\n"
        f"- 총 RAM: `{preset['total_gb']:.1f} GB`\n"
        f"- 가용 RAM: `{preset['available_gb']:.1f} GB`\n"
        f"- 선택된 프리셋: **{preset['tier']}** "
        f"(resolution=`{preset['resolution']}`, chunk_size=`{preset['chunk_size']}`)"
    )

    if isinstance(input_data, Image.Image):
        pil_image = input_data
    else:
        pil_image = Image.fromarray(input_data, mode="RGBA")

    pil_image = pil_image.resize((512, 512), Image.LANCZOS)
    background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
    background.paste(pil_image, mask=pil_image.split()[3])
    input_image = background.convert("RGB")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        model.renderer.set_chunk_size(preset["chunk_size"])
        scene_codes = model([input_image], device=device)

    del input_image, background, pil_image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mesh_chunk = max(512, preset["chunk_size"] // 4)
    model.renderer.set_chunk_size(mesh_chunk)
    use_vertex_color = preset["tier"] not in ("Minimal", "Low")

    try:
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=use_vertex_color,
            resolution=preset["resolution"],
        )
    except (MemoryError, RuntimeError) as e:
        st.warning(f"⚠️ 메모리 부족. 더 낮은 해상도로 재시도합니다... ({e})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fallback_res = max(32, preset["resolution"] // 2)
        model.renderer.set_chunk_size(512)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=fallback_res)
        st.info(f"✅ 폴백 모드로 메쉬 생성 완료 (resolution=`{fallback_res}`)")

    mesh = meshes[0]
    del scene_codes, meshes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timestamp = int(time.time())
    obj_path = os.path.normpath(os.path.join(output_dir, f"mesh_{timestamp}.obj"))
    mesh.export(obj_path)
    return obj_path


# =========================================================
# Streamlit UI
# =========================================================

with st.sidebar:
    st.subheader("🖥️ 시스템 상태")
    _vm = psutil.virtual_memory()
    st.metric("총 RAM", f"{_vm.total / 1e9:.1f} GB")
    st.metric("가용 RAM", f"{_vm.available / 1e9:.1f} GB",
              delta=f"{_vm.percent}% 사용 중", delta_color="inverse")
    _preset = get_inference_preset()
    st.caption(f"예상 프리셋: **{_preset['tier']}**")
    st.caption(f"resolution=`{_preset['resolution']}`, chunk_size=`{_preset['chunk_size']}`")
    st.caption("💡 가용 RAM이 적으면 브라우저/IDE를 닫고 새로고침하세요.")
    st.divider()
    st.caption("3D 뷰어는 별도 터미널에서 실행:")
    st.code("streamlit run viewer.py --server.port 8502")

st.title("🧊 Image to 3D Viewer")
st.write("2D 이미지를 업로드하면 SAM 2로 객체를 추출하고 TripoSR을 통해 3D 모델로 변환합니다.")

st.subheader("Step 1: 이미지 업로드")
uploaded_file = st.file_uploader("2D 이미지 파일(jpg, png)을 선택하세요.", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.sam2_done      = False
        st.session_state.triposr_done   = False
        st.session_state.extracted_image = None
        st.session_state.mesh_path      = None
        st.session_state.last_uploaded  = uploaded_file.name

    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 원본 이미지", use_column_width=True)
    st.divider()

    # --- Step 2: SAM 2 ---
    st.subheader("Step 2: 배경 제거 및 객체 추출 (SAM 2)")

    if st.button("SAM 2 실행 (객체 추출)") or st.session_state.sam2_done:
        if not st.session_state.sam2_done:
            with st.spinner("SAM 2 모델을 로드하고 이미지를 분석하고 있습니다..."):
                predictor, device = load_sam2_model()
                img_rgb = np.array(image.convert("RGB"))
                predictor.set_image(img_rgb)
                h, w, _ = img_rgb.shape
                input_box = np.array([[int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)]])
                masks, scores, logits = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=input_box, multimask_output=False,
                )
                mask_2d = masks.squeeze()
                img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                img_rgba[:, :, :3] = img_rgb
                img_rgba[:, :, 3] = (mask_2d > 0).astype(np.uint8) * 255
                st.session_state.extracted_image = Image.fromarray(img_rgba, "RGBA")
                st.session_state.sam2_done = True

        st.success("✅ 객체 추출 완료!")
        st.image(st.session_state.extracted_image, caption="배경이 제거된 객체 (RGBA)", use_column_width=True)
        st.divider()

        # --- Step 3: TripoSR ---
        st.subheader("Step 3: 3D 메쉬 생성 (TripoSR)")

        if st.button("TripoSR 실행 (3D 변환)") or st.session_state.triposr_done:
            if not st.session_state.triposr_done:
                with st.spinner("TripoSR 모델을 로드하는 중입니다..."):
                    load_triposr_model()
                with st.spinner("🔄 3D 메쉬를 생성하는 중입니다... (환경에 따라 30초~5분 소요)"):
                    mesh_path = run_triposr_inference(st.session_state.extracted_image)
                st.session_state.mesh_path    = mesh_path
                st.session_state.triposr_done = True

            st.success("✅ 3D 메쉬 생성 완료!")

            obj_path = st.session_state.mesh_path
            if obj_path and os.path.exists(obj_path):
                # .obj 다운로드
                with open(obj_path, "rb") as f:
                    st.download_button(
                        label="⬇️ .obj 다운로드",
                        data=f.read(),
                        file_name=os.path.basename(obj_path),
                        mime="model/obj",
                        key="dl_obj",
                    )

                st.divider()

                # 3D 뷰어 링크 (viewer.py 별도 실행)
                st.markdown("#### 🖥️ 3D 뷰어")
                viewer_url = f"http://localhost:8502/?obj={obj_path.replace(os.sep, '/')}"
                st.markdown(
                    f"""
                    <a href="{viewer_url}" target="_blank">
                        <button style="
                            background:#4a4aff; color:white; border:none;
                            padding:10px 24px; border-radius:8px;
                            font-size:15px; cursor:pointer;">
                            🌐 3D 뷰어 새 탭으로 열기
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("※ 3D 뷰어가 안 열리면 별도 터미널에서 `streamlit run viewer.py --server.port 8502` 를 먼저 실행하세요.")
            else:
                st.error("메쉬 파일을 찾을 수 없습니다. 다시 시도해 주세요.")