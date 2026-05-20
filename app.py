import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import torch
import os
import urllib.request
import base64

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
    """
    TripoSR 모델을 로드합니다.
    가중치가 없을 경우 Hugging Face Hub에서 자동으로 다운로드합니다.
    """
    # TripoSR 라이브러리 임포트 (pip install tsr 또는 로컬 설치 가정)
    from tsr.system import TSR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.normpath(os.path.join(base_dir, "checkpoints"))
    triposr_checkpoint = os.path.normpath(os.path.join(checkpoint_dir, "triposr"))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 가중치가 없으면 Hugging Face Hub에서 자동 다운로드
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
# 3D 변환 실행 함수
# =========================================================

def run_triposr_inference(input_data) -> str:
    """
    이미지 데이터를 받아 TripoSR로 3D 메쉬를 생성하고 .obj 파일 경로를 반환합니다.
    """
    model, device = load_triposr_model()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    os.makedirs(output_dir, exist_ok=True)

    # 전달받은 데이터가 이미 PIL Image인지, 순수 배열인지 확인하고 처리합니다.
    if isinstance(input_data, Image.Image):
        pil_image = input_data
    else:
        pil_image = Image.fromarray(input_data, mode="RGBA")

    # TripoSR 전처리: 512×512 리사이즈 후 배경을 흰색으로 합성
    pil_image = pil_image.resize((512, 512), Image.LANCZOS)
    background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
    background.paste(pil_image, mask=pil_image.split()[3])  # alpha 채널을 마스크로 사용
    input_image = background.convert("RGB")

    with torch.no_grad():
        scene_codes = model([input_image], device=device)

    # 메쉬 추출 (marching cubes 해상도: 128)
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=128)
    mesh = meshes[0]

    # .obj 파일 저장
    timestamp = int(time.time())
    obj_filename = f"mesh_{timestamp}.obj"
    obj_path = os.path.normpath(os.path.join(output_dir, obj_filename))
    mesh.export(obj_path)

    return obj_path


# =========================================================
# 3D 뷰어 렌더링 함수 (Three.js 인라인 HTML)
# =========================================================

def render_3d_viewer(obj_path: str):
    """
    .obj 파일을 읽어 Three.js 기반 인라인 3D 뷰어를 Streamlit에 렌더링합니다.
    외부 라이브러리 설치 없이 st.components.v1.html 만으로 동작합니다.

    Args:
        obj_path: 로컬 .obj 파일의 절대 경로
    """
    import streamlit.components.v1 as components

    # .obj 파일을 읽어 Base64로 인코딩 (브라우저로 안전하게 전달)
    with open(obj_path, "r", encoding="utf-8") as f:
        obj_content = f.read()

    obj_b64 = base64.b64encode(obj_content.encode("utf-8")).decode("utf-8")

    html_code = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0d0d14; overflow: hidden; }}
    #canvas-container {{
      width: 100%;
      height: 480px;
      position: relative;
    }}
    canvas {{ display: block; width: 100% !important; height: 100% !important; }}
    #controls-hint {{
      position: absolute;
      bottom: 12px;
      left: 50%;
      transform: translateX(-50%);
      color: rgba(180, 180, 220, 0.6);
      font-family: 'Segoe UI', sans-serif;
      font-size: 11px;
      letter-spacing: 0.5px;
      pointer-events: none;
      white-space: nowrap;
    }}
    #loading-overlay {{
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: #0d0d14;
      color: #8888cc;
      font-family: 'Segoe UI', sans-serif;
      font-size: 13px;
      gap: 16px;
      z-index: 10;
      transition: opacity 0.5s ease;
    }}
    .spinner {{
      width: 36px;
      height: 36px;
      border: 3px solid rgba(100, 100, 200, 0.2);
      border-top-color: #8888ff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
<div id="canvas-container">
  <div id="loading-overlay">
    <div class="spinner"></div>
    <span>3D 메쉬 불러오는 중…</span>
  </div>
  <canvas id="three-canvas"></canvas>
  <div id="controls-hint">🖱 드래그: 회전 &nbsp;|&nbsp; 스크롤: 줌 &nbsp;|&nbsp; 우클릭 드래그: 이동</div>
</div>

<!-- Three.js r128 CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
// ── Base64 디코딩 ──────────────────────────────────────────
const OBJ_B64 = "{obj_b64}";
const objText = decodeURIComponent(escape(atob(OBJ_B64)));

// ── Scene 셋업 ─────────────────────────────────────────────
const container = document.getElementById('canvas-container');
const canvas    = document.getElementById('three-canvas');
const W = container.clientWidth;
const H = 480;

const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
renderer.setSize(W, H);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d14);

const camera = new THREE.PerspectiveCamera(45, W / H, 0.01, 1000);
camera.position.set(0, 0.5, 2.5);

// ── 조명 ──────────────────────────────────────────────────
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(5, 8, 5);
dirLight.castShadow = true;
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0x8888ff, 0.4);
fillLight.position.set(-5, 2, -5);
scene.add(fillLight);

// ── OBJ 파서 (Three.js 빌트인 없이 직접 구현) ─────────────
function parseOBJ(text) {{
  const positions = [];
  const normals   = [];
  const uvs       = [];
  const verts     = [];

  for (const rawLine of text.split('\\n')) {{
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\\s+/);
    switch (parts[0]) {{
      case 'v':
        positions.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
        break;
      case 'vn':
        normals.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
        break;
      case 'vt':
        uvs.push(parseFloat(parts[1]), parseFloat(parts[2]));
        break;
      case 'f':
        const faceVerts = parts.slice(1);
        // 삼각형 팬 분할 (n-gon 지원)
        for (let i = 1; i < faceVerts.length - 1; i++) {{
          [faceVerts[0], faceVerts[i], faceVerts[i + 1]].forEach(token => {{
            const [vi, ti, ni] = token.split('/').map(x => x ? parseInt(x) - 1 : undefined);
            verts.push({{ vi, ti, ni }});
          }});
        }}
        break;
    }}
  }}

  const posArr = new Float32Array(verts.length * 3);
  const nrmArr = normals.length ? new Float32Array(verts.length * 3) : null;
  const uvArr  = uvs.length    ? new Float32Array(verts.length * 2) : null;

  verts.forEach((v, i) => {{
    posArr[i*3]   = positions[v.vi*3];
    posArr[i*3+1] = positions[v.vi*3+1];
    posArr[i*3+2] = positions[v.vi*3+2];
    if (nrmArr && v.ni !== undefined) {{
      nrmArr[i*3]   = normals[v.ni*3];
      nrmArr[i*3+1] = normals[v.ni*3+1];
      nrmArr[i*3+2] = normals[v.ni*3+2];
    }}
    if (uvArr && v.ti !== undefined) {{
      uvArr[i*2]   = uvs[v.ti*2];
      uvArr[i*2+1] = uvs[v.ti*2+1];
    }}
  }});

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  if (nrmArr) geo.setAttribute('normal', new THREE.BufferAttribute(nrmArr, 3));
  else geo.computeVertexNormals();
  if (uvArr)  geo.setAttribute('uv', new THREE.BufferAttribute(uvArr, 2));
  return geo;
}}

// ── 메쉬 생성 ──────────────────────────────────────────────
const geo = parseOBJ(objText);

// 중심 정렬 및 정규화 (크기를 뷰포트에 맞게 조정)
geo.computeBoundingBox();
const box    = geo.boundingBox;
const center = new THREE.Vector3();
box.getCenter(center);
geo.translate(-center.x, -center.y, -center.z);

const size   = new THREE.Vector3();
box.getSize(size);
const maxDim = Math.max(size.x, size.y, size.z);
const scale  = 1.8 / maxDim;
geo.scale(scale, scale, scale);

const mat = new THREE.MeshStandardMaterial({{
  color: 0xccccee,
  metalness: 0.15,
  roughness: 0.6,
  side: THREE.DoubleSide,
}});
const mesh = new THREE.Mesh(geo, mat);
mesh.castShadow    = true;
mesh.receiveShadow = true;
scene.add(mesh);

// 격자 바닥
const gridHelper = new THREE.GridHelper(4, 20, 0x333366, 0x222244);
gridHelper.position.y = -1.0;
scene.add(gridHelper);

// ── 로딩 오버레이 제거 ─────────────────────────────────────
const overlay = document.getElementById('loading-overlay');
overlay.style.opacity = '0';
setTimeout(() => overlay.style.display = 'none', 500);

// ── 마우스 오빗 컨트롤 (직접 구현) ───────────────────────
let isDragging  = false;
let isRightDrag = false;
let prevX = 0, prevY = 0;
let rotX = 0, rotY = 0;
let panX = 0, panY = 0;
let zoom = 2.5;

canvas.addEventListener('mousedown', e => {{
  isDragging  = true;
  isRightDrag = (e.button === 2);
  prevX = e.clientX;
  prevY = e.clientY;
}});
window.addEventListener('mouseup', () => {{ isDragging = false; }});
window.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  const dx = e.clientX - prevX;
  const dy = e.clientY - prevY;
  prevX = e.clientX;
  prevY = e.clientY;
  if (isRightDrag) {{
    panX += dx * 0.004;
    panY -= dy * 0.004;
  }} else {{
    rotY += dx * 0.6;
    rotX += dy * 0.6;
  }}
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom = Math.max(0.5, Math.min(10, zoom + e.deltaY * 0.005));
}}, {{ passive: false }});
canvas.addEventListener('contextmenu', e => e.preventDefault());

// ── 자동 회전 ─────────────────────────────────────────────
let autoRotate = true;
canvas.addEventListener('mousedown', () => {{ autoRotate = false; }});

// ── 렌더 루프 ─────────────────────────────────────────────
const clock = new THREE.Clock();
function animate() {{
  requestAnimationFrame(animate);
  const dt = clock.getDelta();

  if (autoRotate) rotY += dt * 25;

  mesh.rotation.x = THREE.MathUtils.degToRad(rotX);
  mesh.rotation.y = THREE.MathUtils.degToRad(rotY);

  camera.position.set(
    panX,
    0.5 + panY,
    zoom
  );
  camera.lookAt(panX, panY, 0);

  renderer.render(scene, camera);
}}
animate();

// ── 리사이즈 대응 ──────────────────────────────────────────
window.addEventListener('resize', () => {{
  const w = container.clientWidth;
  renderer.setSize(w, H);
  camera.aspect = w / H;
  camera.updateProjectionMatrix();
}});
</script>
</body>
</html>
"""
    components.html(html_code, height=500)


# =========================================================
# Streamlit UI
# =========================================================

st.title("🧊 Image to 3D Viewer")
st.write("2D 이미지를 업로드하면 SAM 2로 객체를 추출하고 TripoSR을 통해 3D 모델로 변환합니다.")

# --- Step 1: 이미지 업로드 ---
st.subheader("Step 1: 이미지 업로드")
uploaded_file = st.file_uploader("2D 이미지 파일(jpg, png)을 선택하세요.", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 새 파일 업로드 시 세션 리셋
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.sam2_done    = False
        st.session_state.triposr_done = False
        st.session_state.extracted_image = None
        st.session_state.mesh_path    = None
        st.session_state.last_uploaded = uploaded_file.name

    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 원본 이미지", use_column_width=True)
    st.divider()

    # --- Step 2: SAM 2 객체 추출 ---
    st.subheader("Step 2: 배경 제거 및 객체 추출 (SAM 2)")

    if st.button("SAM 2 실행 (객체 추출)") or st.session_state.sam2_done:
        if not st.session_state.sam2_done:
            with st.spinner("SAM 2 모델을 로드하고 이미지를 분석하고 있습니다..."):
                predictor, device = load_sam2_model()

                img_rgb = np.array(image.convert("RGB"))
                predictor.set_image(img_rgb)

                h, w, _ = img_rgb.shape
                
                # 💡 [해결 포인트] 점(Point) 대신 박스(Box) 프롬프트 사용
                # 캔이 중앙에 있으므로 이미지 상하좌우 15%~85% 영역에 가상의 네모 박스를 칩니다.
                # 에러 방지를 위해 반드시 정수(int)로 변환해야 합니다.
                input_box = np.array([[
                    int(w * 0.15), 
                    int(h * 0.15), 
                    int(w * 0.85), 
                    int(h * 0.85)
                ]])

                # predict 함수에 점 대신 박스 좌표를 넣습니다.
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )

                # 마스크 처리 및 RGBA 변환
                mask_2d = masks.squeeze()
                img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                img_rgba[:, :, :3] = img_rgb
                
                # 배경은 투명(0), 객체는 불투명(255)으로 설정
                img_rgba[:, :, 3] = (mask_2d > 0).astype(np.uint8) * 255

                # Streamlit 렌더링 에러 방지를 위해 PIL Image 객체로 변환하여 저장
                final_pil_image = Image.fromarray(img_rgba, "RGBA")
                st.session_state.extracted_image = final_pil_image
                st.session_state.sam2_done = True

        st.success("✅ 객체 추출 완료!")
        if st.session_state.extracted_image is not None:
            st.image(st.session_state.extracted_image, caption="배경이 제거된 객체 (RGBA)", use_column_width=True)
        st.divider()

        # --- Step 3: TripoSR 3D 변환 ---
        st.subheader("Step 3: 3D 메쉬 생성 및 뷰어 (TripoSR)")

        if st.button("TripoSR 실행 (3D 변환)") or st.session_state.triposr_done:
            if not st.session_state.triposr_done:
                # ① 모델 로드 (캐시 적용)
                with st.spinner("TripoSR 모델을 로드하는 중입니다... (최초 1회 가중치 다운로드 약 2GB)"):
                    load_triposr_model()

                # ② 3D 추론 실행
                with st.spinner("🔄 3D 메쉬를 생성하는 중입니다... (GPU 기준 약 30초~2분 소요)"):
                    mesh_path = run_triposr_inference(st.session_state.extracted_image)
                    st.session_state.mesh_path    = mesh_path
                    st.session_state.triposr_done = True

            st.success(f"✅ 3D 메쉬 생성 완료!")

            # ③ .obj 파일 다운로드 버튼
            obj_path = st.session_state.mesh_path
            if obj_path and os.path.exists(obj_path):
                with open(obj_path, "rb") as f:
                    obj_bytes = f.read()

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"저장 경로: `{obj_path}`")
                with col2:
                    st.download_button(
                        label="⬇️ .obj 다운로드",
                        data=obj_bytes,
                        file_name=os.path.basename(obj_path),
                        mime="model/obj",
                    )

                # ④ 인라인 3D 뷰어 렌더링
                st.markdown("#### 🖥️ 3D 뷰어 (Three.js)")
                st.caption("드래그로 회전 | 스크롤로 줌 | 우클릭 드래그로 이동 | 처음 3초간 자동 회전")
                render_3d_viewer(obj_path)
            else:
                st.error("메쉬 파일을 찾을 수 없습니다. 다시 시도해 주세요.")