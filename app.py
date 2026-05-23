import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import torch
import os
import urllib.request
import base64
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
if 'viewer_html_bytes' not in st.session_state:
    st.session_state.viewer_html_bytes = None
if 'viewer_filename' not in st.session_state:
    st.session_state.viewer_filename = None


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
    """
    사용 가능한 RAM을 기준으로 안전한 resolution과 chunk_size를 반환.
    available 메모리에서 안전 마진 1.5GB 빼고 계산.
    """
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)

    # 안전 마진 확보 (OS / 브라우저 / Streamlit 자체)
    usable_gb = max(0.5, available_gb - 1.5)

    if usable_gb < 2.5:
        preset = {"resolution": 32, "chunk_size": 1024, "tier": "Minimal"}
    elif usable_gb < 6:
        preset = {"resolution": 64, "chunk_size": 4096, "tier": "Low"}
    elif usable_gb < 14:
        preset = {"resolution": 128, "chunk_size": 16384, "tier": "Medium"}
    elif usable_gb < 30:
        preset = {"resolution": 192, "chunk_size": 65536, "tier": "High"}
    else:
        preset = {"resolution": 256, "chunk_size": 131072, "tier": "Ultra"}

    preset["available_gb"] = available_gb
    preset["total_gb"] = total_gb
    preset["usable_gb"] = usable_gb
    return preset


# =========================================================
# 3D 변환 실행 함수 (자동 프리셋 + 단계별 메모리 해제 + 폴백)
# =========================================================

def run_triposr_inference(input_data) -> str:
    model, device = load_triposr_model()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    os.makedirs(output_dir, exist_ok=True)

    # ───── 1. 환경 분석 및 프리셋 선택 ─────
    preset = get_inference_preset()
    st.info(
        f"🖥️ **자동 환경 분석**\n\n"
        f"- 총 RAM: `{preset['total_gb']:.1f} GB`\n"
        f"- 가용 RAM: `{preset['available_gb']:.1f} GB`\n"
        f"- 선택된 프리셋: **{preset['tier']}** "
        f"(resolution=`{preset['resolution']}`, chunk_size=`{preset['chunk_size']}`)"
    )

    # ───── 2. 이미지 전처리 ─────
    if isinstance(input_data, Image.Image):
        pil_image = input_data
    else:
        pil_image = Image.fromarray(input_data, mode="RGBA")

    pil_image = pil_image.resize((512, 512), Image.LANCZOS)
    background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
    background.paste(pil_image, mask=pil_image.split()[3])
    input_image = background.convert("RGB")

    # ───── 3. 추론 전 메모리 정리 ─────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ───── 4. Scene codes 추출 ─────
    with torch.no_grad():
        model.renderer.set_chunk_size(preset["chunk_size"])
        scene_codes = model([input_image], device=device)

    # 더 이상 안 쓰는 변수 즉시 해제
    del input_image, background, pil_image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ───── 5. Mesh 추출 (chunk_size를 더 작게 강제) ─────
    # extract_mesh는 resolution³ 만큼의 포인트를 처리하므로
    # 더 보수적인 chunk_size로 한 번 더 조임
    mesh_chunk = max(512, preset["chunk_size"] // 4)
    model.renderer.set_chunk_size(mesh_chunk)

    # 저메모리 환경에서는 vertex color 끔 (RAM 30~40% 추가 절약)
    use_vertex_color = preset["tier"] not in ("Minimal", "Low")

    try:
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=use_vertex_color,
            resolution=preset["resolution"],
        )
    except (MemoryError, RuntimeError) as e:
        # 그래도 OOM 나면 더 낮춰서 재시도
        st.warning(f"⚠️ 메모리 부족 감지. 더 낮은 해상도로 재시도합니다... ({e})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fallback_res = max(32, preset["resolution"] // 2)
        model.renderer.set_chunk_size(512)
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=False,
            resolution=fallback_res,
        )
        st.info(f"✅ 폴백 모드로 메쉬 생성 완료 (resolution=`{fallback_res}`)")

    mesh = meshes[0]

    # ───── 6. 최종 정리 ─────
    del scene_codes, meshes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ───── 7. 저장 ─────
    timestamp = int(time.time())
    obj_filename = f"mesh_{timestamp}.obj"
    obj_path = os.path.normpath(os.path.join(output_dir, obj_filename))
    mesh.export(obj_path)

    return obj_path


# =========================================================
# 3D 뷰어 HTML 생성 함수 (st. 코드 없음 → rerun 충돌 없음)
# =========================================================

def build_viewer_html(obj_path: str) -> bytes:
    with open(obj_path, "r", encoding="utf-8") as f:
        obj_content = f.read()

    obj_b64 = base64.b64encode(obj_content.encode("utf-8")).decode("utf-8")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <title>3D Viewer</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:100%; background:#0d0d14; overflow:hidden; }}
    canvas {{ display:block; width:100vw; height:100vh; }}
    #hint {{
      position:fixed; bottom:12px; left:50%; transform:translateX(-50%);
      color:rgba(180,180,220,0.6); font:11px 'Segoe UI',sans-serif;
      pointer-events:none; white-space:nowrap; z-index:5;
    }}
    #loading {{
      position:fixed; inset:0; display:flex; flex-direction:column;
      align-items:center; justify-content:center;
      background:#0d0d14; color:#8888cc; font:14px 'Segoe UI',sans-serif;
      gap:16px; z-index:10; transition:opacity 0.5s;
    }}
    .spinner {{
      width:40px; height:40px;
      border:3px solid rgba(100,100,200,0.2);
      border-top-color:#8888ff; border-radius:50%;
      animation:spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
  </style>
</head>
<body>
<div id="loading"><div class="spinner"></div><span>3D 메쉬 불러오는 중…</span></div>
<canvas id="c"></canvas>
<div id="hint">🖱 드래그: 회전 &nbsp;|&nbsp; 스크롤: 줌 &nbsp;|&nbsp; 우클릭 드래그: 이동</div>
<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}}}
</script>
<script type="module">
import * as THREE from 'three';

const OBJ_B64 = "{obj_b64}";
const objText = new TextDecoder().decode(Uint8Array.from(atob(OBJ_B64), c=>c.charCodeAt(0)));

function parseOBJ(text) {{
  const positions=[], normals=[], uvs=[], verts=[];
  for (const raw of text.split('\\n')) {{
    const line=raw.trim();
    if (!line || line.startsWith('#')) continue;
    const p=line.split(/\\s+/);
    if      (p[0]==='v')  positions.push(+p[1],+p[2],+p[3]);
    else if (p[0]==='vn') normals.push(+p[1],+p[2],+p[3]);
    else if (p[0]==='vt') uvs.push(+p[1],+p[2]);
    else if (p[0]==='f') {{
      const fv=p.slice(1);
      for (let i=1;i<fv.length-1;i++)
        [fv[0],fv[i],fv[i+1]].forEach(t=>{{
          const [vi,ti,ni]=t.split('/').map(x=>x?+x-1:undefined);
          verts.push({{vi,ti,ni}});
        }});
    }}
  }}
  const posArr=new Float32Array(verts.length*3);
  const nrmArr=normals.length?new Float32Array(verts.length*3):null;
  const uvArr=uvs.length?new Float32Array(verts.length*2):null;
  verts.forEach((v,i)=>{{
    posArr[i*3]=positions[v.vi*3]; posArr[i*3+1]=positions[v.vi*3+1]; posArr[i*3+2]=positions[v.vi*3+2];
    if(nrmArr&&v.ni!==undefined){{nrmArr[i*3]=normals[v.ni*3];nrmArr[i*3+1]=normals[v.ni*3+1];nrmArr[i*3+2]=normals[v.ni*3+2];}}
    if(uvArr&&v.ti!==undefined){{uvArr[i*2]=uvs[v.ti*2];uvArr[i*2+1]=uvs[v.ti*2+1];}}
  }});
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(posArr,3));
  if(nrmArr) geo.setAttribute('normal',new THREE.BufferAttribute(nrmArr,3));
  else geo.computeVertexNormals();
  if(uvArr) geo.setAttribute('uv',new THREE.BufferAttribute(uvArr,2));
  return geo;
}}

const canvas=document.getElementById('c');
const renderer=new THREE.WebGLRenderer({{canvas,antialias:true}});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(window.innerWidth,window.innerHeight);
renderer.shadowMap.enabled=true;

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0d0d14);
const camera=new THREE.PerspectiveCamera(45,window.innerWidth/window.innerHeight,0.01,1000);
camera.position.set(0,0.5,2.5);

scene.add(new THREE.AmbientLight(0xffffff,0.4));
const dir=new THREE.DirectionalLight(0xffffff,1.2);
dir.position.set(5,8,5); dir.castShadow=true; scene.add(dir);
const fill=new THREE.DirectionalLight(0x8888ff,0.4);
fill.position.set(-5,2,-5); scene.add(fill);

const geo=parseOBJ(objText);
geo.computeBoundingBox();
const ctr=new THREE.Vector3(); geo.boundingBox.getCenter(ctr);
geo.translate(-ctr.x,-ctr.y,-ctr.z);
const sz=new THREE.Vector3(); geo.boundingBox.getSize(sz);
geo.scale(...Array(3).fill(1.8/Math.max(sz.x,sz.y,sz.z)));

const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({{color:0xccccee,metalness:0.15,roughness:0.6,side:THREE.DoubleSide}}));
mesh.castShadow=mesh.receiveShadow=true;
scene.add(mesh);
const grid=new THREE.GridHelper(4,20,0x333366,0x222244);
grid.position.y=-1; scene.add(grid);

document.getElementById('loading').style.opacity='0';
setTimeout(()=>document.getElementById('loading').remove(),500);

let drag=false,rDrag=false,px=0,py=0,rotX=0,rotY=0,panX=0,panY=0,zoom=2.5,auto=true;
canvas.addEventListener('mousedown',e=>{{drag=true;rDrag=e.button===2;px=e.clientX;py=e.clientY;auto=false;}});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{{
  if(!drag)return;
  const dx=e.clientX-px,dy=e.clientY-py;px=e.clientX;py=e.clientY;
  if(rDrag){{panX+=dx*0.004;panY-=dy*0.004;}}else{{rotY+=dx*0.6;rotX+=dy*0.6;}}
}});
canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(0.5,Math.min(10,zoom+e.deltaY*0.005));}},{{passive:false}});
canvas.addEventListener('contextmenu',e=>e.preventDefault());

const clock=new THREE.Clock();
(function animate(){{
  requestAnimationFrame(animate);
  const dt=clock.getDelta();
  if(auto) rotY+=dt*25;
  mesh.rotation.set(THREE.MathUtils.degToRad(rotX),THREE.MathUtils.degToRad(rotY),0);
  camera.position.set(panX,0.5+panY,zoom);
  camera.lookAt(panX,panY,0);
  renderer.render(scene,camera);
}})();

window.addEventListener('resize',()=>{{
  renderer.setSize(window.innerWidth,window.innerHeight);
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
}});
</script>
</body>
</html>"""

    return html.encode("utf-8")


# =========================================================
# Streamlit UI
# =========================================================

# --- 사이드바: 시스템 상태 ---
with st.sidebar:
    st.subheader("🖥️ 시스템 상태")
    _vm = psutil.virtual_memory()
    st.metric("총 RAM", f"{_vm.total / 1e9:.1f} GB")
    st.metric(
        "가용 RAM",
        f"{_vm.available / 1e9:.1f} GB",
        delta=f"{_vm.percent}% 사용 중",
        delta_color="inverse",
    )
    _preset = get_inference_preset()
    st.caption(f"예상 프리셋: **{_preset['tier']}**")
    st.caption(
        f"resolution=`{_preset['resolution']}`, "
        f"chunk_size=`{_preset['chunk_size']}`"
    )
    st.caption("💡 가용 RAM이 적으면 브라우저/IDE를 닫고 새로고침하세요.")

st.title("🧊 Image to 3D Viewer")
st.write("2D 이미지를 업로드하면 SAM 2로 객체를 추출하고 TripoSR을 통해 3D 모델로 변환합니다.")

# --- Step 1: 이미지 업로드 ---
st.subheader("Step 1: 이미지 업로드")
uploaded_file = st.file_uploader("2D 이미지 파일(jpg, png)을 선택하세요.", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.sam2_done         = False
        st.session_state.triposr_done      = False
        st.session_state.extracted_image   = None
        st.session_state.mesh_path         = None
        st.session_state.viewer_html_bytes = None
        st.session_state.viewer_filename   = None
        st.session_state.last_uploaded     = uploaded_file.name

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

                input_box = np.array([[int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)]])
                masks, scores, logits = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=input_box, multimask_output=False,
                )

                mask_2d = masks.squeeze()
                img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                img_rgba[:, :, :3] = img_rgb
                img_rgba[:, :, 3] = (mask_2d > 0).astype(np.uint8) * 255

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
                with st.spinner("TripoSR 모델을 로드하는 중입니다..."):
                    load_triposr_model()

                with st.spinner("🔄 3D 메쉬를 생성하는 중입니다... (환경에 따라 30초~5분 소요)"):
                    mesh_path = run_triposr_inference(st.session_state.extracted_image)

                with st.spinner("🖼️ 뷰어 HTML을 생성하는 중입니다..."):
                    viewer_bytes = build_viewer_html(mesh_path)
                    viewer_fname = os.path.basename(mesh_path).replace(".obj", "_viewer.html")

                # [문제2 수정] 모든 결과를 한꺼번에 세션에 저장 후 triposr_done = True
                # triposr_done을 마지막에 True로 세팅해야 rerun 후 버튼이 정상 렌더링됨
                st.session_state.mesh_path         = mesh_path
                st.session_state.viewer_html_bytes = viewer_bytes
                st.session_state.viewer_filename   = viewer_fname
                st.session_state.triposr_done      = True
                st.rerun()  # 세션 저장 완료 후 명시적 rerun으로 UI 갱신

            # --- triposr_done == True 일 때 항상 렌더링되는 영역 ---
            st.success("✅ 3D 메쉬 생성 완료!")

            obj_path = st.session_state.mesh_path
            if obj_path and os.path.exists(obj_path):

                # ③ .obj 다운로드
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
                        key="dl_obj",
                    )

                st.divider()

                # ④ 3D 뷰어 HTML 다운로드
                st.markdown("#### 🖥️ 3D 뷰어")
                st.info("⬇️ 버튼으로 뷰어 파일을 다운로드한 뒤, 파일을 더블클릭하면 브라우저에서 3D 모델을 바로 볼 수 있습니다.")
                st.download_button(
                    label="🌐 3D 뷰어 HTML 다운로드",
                    data=st.session_state.viewer_html_bytes,
                    file_name=st.session_state.viewer_filename,
                    mime="text/html",
                    key="dl_viewer",
                )

            else:
                st.error("메쉬 파일을 찾을 수 없습니다. 다시 시도해 주세요.")