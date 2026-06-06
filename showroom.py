import streamlit as st
import streamlit.components.v1 as components
import os
import json
import base64
import glob

st.set_page_config(page_title="3D Showroom & Viewer", page_icon="🏠", layout="wide")

# =========================================================
# 공통 유틸리티 & JS 파서 
# =========================================================
JS_PARSE_OBJ = r"""
function parseOBJ(text) {
    const positions=[], normals=[], uvs=[], verts=[];
    for (const raw of text.split('\n')) {
        const line=raw.trim();
        if (!line || line.startsWith('#')) continue;
        const p=line.split(/\s+/);
        if (p[0]==='v') positions.push(+p[1],+p[2],+p[3]);
        else if (p[0]==='vn') normals.push(+p[1],+p[2],+p[3]);
        else if (p[0]==='f') {
            const fv=p.slice(1);
            for (let i=1;i<fv.length-1;i++) {
                [fv[0],fv[i],fv[i+1]].forEach(t=>{
                    const [vi,ti,ni]=t.split('/').map(x=>x?+x-1:undefined);
                    verts.push({vi,ti,ni});
                });
            }
        }
    }
    const posArr=new Float32Array(verts.length*3);
    verts.forEach((v,i)=>{
        posArr[i*3]=positions[v.vi*3];
        posArr[i*3+1]=positions[v.vi*3+1];
        posArr[i*3+2]=positions[v.vi*3+2];
    });
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.BufferAttribute(posArr,3));
    geo.computeVertexNormals();
    return geo;
}
"""

def get_obj_list() -> list:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    if not os.path.exists(mesh_dir): return []
    return sorted(glob.glob(os.path.join(mesh_dir, "*.obj")), reverse=True)

def read_obj_b64(obj_path: str) -> str:
    with open(obj_path, "r", encoding="utf-8") as f: content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("utf-8")

params = st.query_params
single_obj_path = params.get("obj", "")

is_single = bool(single_obj_path)

if is_single:
    single_obj_path = single_obj_path.replace("/", os.sep)
    if not os.path.exists(single_obj_path): st.stop()
    st.title("🧊 3D Single Viewer")
    obj_data = [{"id": 0, "name": "single", "b64": read_obj_b64(single_obj_path), "x": 0, "z": 0}]
else:
    st.title("🏠 3D Virtual Showroom")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    obj_files = get_obj_list()
    
    selected = []
    cols = st.columns(4)
    for i, f in enumerate(obj_files):
        if cols[i % 4].checkbox(os.path.basename(f), key=f"chk_{i}", value=(i < 3)):
            selected.append(f)
            
    if not selected: st.stop()
    obj_data = [{"id": i, "name": os.path.basename(p), "b64": read_obj_b64(p), "x": (i%3)*2.5-2.5, "z": (i//3)*2.5-2.5} for i, p in enumerate(selected)]

objects_json = json.dumps(obj_data)

# 🔥 사용자 친화적 HTML UI 패널 + HDRI / 컨트롤러 결합
html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:100%; background:#111; overflow:hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    canvas {{ display:block; outline: none; }}
    
    /* 기존 스타일 유지 & 개선된 컨트롤 패널 */
    #ui {{ position:absolute; top:20px; right:20px; width:260px; background:rgba(30,30,45,0.9); border:1px solid rgba(100,100,200,0.3); border-radius:12px; padding:16px; z-index:20; color:#eee; box-shadow: 0 8px 32px rgba(0,0,0,0.8); backdrop-filter: blur(8px); }}
    #ui h3 {{ color:#bbbbff; font-size:14px; margin-bottom:12px; font-weight:600; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 6px; }}
    
    .btn-group {{ display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }}
    button {{ background: #2a2a44; color: #fff; border: 1px solid #445; padding: 10px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: 0.2s; }}
    button:hover {{ background: #3a3a5a; border-color:#668; }}
    button.active {{ background: #4a4aff; border-color: #7777ff; font-weight: bold; box-shadow: 0 0 10px rgba(74,74,255,0.4); }}
    
    .control-group {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .control-group span {{ font-size: 12px; font-weight: 500; color: #ccc; }}
    .control-group input[type=range] {{ width: 120px; cursor: pointer; }}
    .control-group input[type=color] {{ cursor: pointer; width: 40px; height: 24px; border: none; border-radius: 4px; background: transparent; }}
    
    input:disabled {{ opacity: 0.3; cursor: not-allowed; }}
    
    #hint {{ position:fixed; bottom:16px; left:50%; transform:translateX(-50%); color:rgba(220,220,255,0.8); font-size:12px; background:rgba(0,0,0,0.6); padding:10px 20px; border-radius:24px; pointer-events:none; border: 1px solid rgba(255,255,255,0.1); }}
  </style>
</head>
<body>
<canvas id="c"></canvas>

<div id="ui">
    <h3>🖱️ 객체 컨트롤 (W/E/R)</h3>
    <div class="btn-group">
        <button id="btn-t" class="active">✋ 이동 (W)</button>
        <button id="btn-s">🔍 크기 (E)</button>
        <button id="btn-r">🔄 회전 (R)</button>
    </div>
    
    <h3>🎨 재질 세팅</h3>
    <div class="control-group">
        <span>색상 (Color)</span>
        <input type="color" id="ctrl-color" value="#cccccc" disabled>
    </div>
    <div class="control-group">
        <span>거칠기 (가죽느낌)</span>
        <input type="range" id="ctrl-rough" min="0" max="1" step="0.05" value="0.5" disabled>
    </div>
    <div class="control-group">
        <span>금속성 (유광느낌)</span>
        <input type="range" id="ctrl-metal" min="0" max="1" step="0.05" value="0.1" disabled>
    </div>
    
    <h3 style="margin-top:16px;">🌍 조명 및 스튜디오</h3>
    <div class="control-group">
        <span>조명 밝기</span>
        <input type="range" id="ctrl-expo" min="0.1" max="3" step="0.1" value="1.0">
    </div>
    <div class="control-group" style="justify-content: flex-start; gap: 10px;">
        <input type="checkbox" id="ctrl-autorotate" {'checked' if is_single else ''}>
        <label for="ctrl-autorotate" style="font-size:12px; color:#ccc; cursor:pointer;">시네마틱 턴테이블</label>
    </div>
    <div class="control-group">
        <span>회전 속도</span>
        <input type="range" id="ctrl-speed" min="0.5" max="10" step="0.5" value="2.0">
    </div>
</div>

<div id="hint">객체 클릭: 선택 | 허공 클릭: 해제 | 휠: 초정밀 줌 | 우클릭 드래그: 화면 이동</div>

<script type="importmap">{{
    "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
    }}
}}</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ TransformControls }} from 'three/addons/controls/TransformControls.js';
import {{ RoomEnvironment }} from 'three/addons/environments/RoomEnvironment.js';

const OBJECTS = {objects_json};
const IS_SINGLE = {'true' if is_single else 'false'};
{JS_PARSE_OBJ}

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene(); 
scene.background = new THREE.Color(0x111111);

// 🌟 고급 HDRI 조명
const pmremGenerator = new THREE.PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 500);
camera.position.set(0, IS_SINGLE ? 1 : 6, IS_SINGLE ? 4 : 14);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.05;
orbit.enableZoom = false; 
orbit.autoRotate = IS_SINGLE; // 단일모드면 자동 켜짐

// 정밀 마우스 줌
canvas.addEventListener('wheel', function(event) {{
    event.preventDefault();
    const distance = camera.position.distanceTo(orbit.target);
    let scale = 1 + (event.deltaY * 0.0015);
    const direction = new THREE.Vector3().subVectors(camera.position, orbit.target).normalize();
    camera.position.copy(orbit.target).addScaledVector(direction, distance * scale);
    orbit.update();
}}, {{ passive: false }});

// 물리 조명 (그림자용)
const dirLight = new THREE.DirectionalLight(0xffffff, 2.0); 
dirLight.position.set(5, 10, 7); 
dirLight.castShadow = true; 
dirLight.shadow.mapSize.width = 2048; 
dirLight.shadow.mapSize.height = 2048;
dirLight.shadow.bias = -0.0001;
scene.add(dirLight);

// 바닥 및 그리드
const floor = new THREE.Mesh(new THREE.PlaneGeometry(100,100), new THREE.MeshStandardMaterial({{color:0x1a1a1a, roughness:0.1, metalness:0.8}}));
floor.rotation.x = -Math.PI/2; floor.position.y = -1; floor.receiveShadow = true; 
scene.add(floor);

const grid = new THREE.GridHelper(100, 100, 0x333333, 0x222222); 
grid.position.y = -0.99; 
scene.add(grid);

const meshes = [];
let activeMesh = null;

// 객체 로드
for (const obj of OBJECTS) {{
    const geo = parseOBJ(new TextDecoder().decode(Uint8Array.from(atob(obj.b64), c=>c.charCodeAt(0))));
    geo.computeBoundingBox();
    const ctr = new THREE.Vector3(); geo.boundingBox.getCenter(ctr); geo.translate(-ctr.x, -ctr.y, -ctr.z);
    const sz = new THREE.Vector3(); geo.boundingBox.getSize(sz); geo.scale(...Array(3).fill(2.0/Math.max(sz.x, sz.y, sz.z)));
    
    const mat = new THREE.MeshStandardMaterial({{ color: 0xcccccc, roughness: 0.5, metalness: 0.1, side: THREE.DoubleSide }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true; 
    mesh.position.set(obj.x, 0, obj.z);
    scene.add(mesh); meshes.push(mesh);
    if(IS_SINGLE) activeMesh = mesh;
}}

// ==========================================
// UI 및 조작(Transform) 로직 바인딩
// ==========================================
const tControl = new TransformControls(camera, renderer.domElement);
tControl.addEventListener('dragging-changed', e => orbit.enabled = !e.value);
scene.add(tControl);

const elColor = document.getElementById('ctrl-color');
const elRough = document.getElementById('ctrl-rough');
const elMetal = document.getElementById('ctrl-metal');
const elExpo = document.getElementById('ctrl-expo');
const elAutoRot = document.getElementById('ctrl-autorotate');
const elSpeed = document.getElementById('ctrl-speed');

// 객체 선택 함수
function selectMesh(mesh) {{
    activeMesh = mesh;
    if (activeMesh && !IS_SINGLE) tControl.attach(activeMesh);
    
    const isActive = !!activeMesh;
    elColor.disabled = !isActive;
    elRough.disabled = !isActive;
    elMetal.disabled = !isActive;
    
    if(isActive) {{
        elColor.value = '#' + activeMesh.material.color.getHexString();
        elRough.value = activeMesh.material.roughness;
        elMetal.value = activeMesh.material.metalness;
    }} else {{
        elColor.value = '#cccccc';
    }}
}}

// 단일 모드일 때 기본 활성화
if (IS_SINGLE && activeMesh) selectMesh(activeMesh);

// 마우스 클릭(Raycaster) 로직
if (!IS_SINGLE) {{
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    
    canvas.addEventListener('pointerdown', e => {{
        if(tControl.dragging) return;
        mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(meshes);
        
        if(intersects.length > 0) selectMesh(intersects[0].object);
        else {{ tControl.detach(); selectMesh(null); }}
    }});
}}

// 재질 슬라이더 이벤트 연결
elColor.addEventListener('input', e => {{ if(activeMesh) activeMesh.material.color.set(e.target.value); }});
elRough.addEventListener('input', e => {{ if(activeMesh) activeMesh.material.roughness = parseFloat(e.target.value); }});
elMetal.addEventListener('input', e => {{ if(activeMesh) activeMesh.material.metalness = parseFloat(e.target.value); }});

// 환경 설정 연결
elExpo.addEventListener('input', e => {{ renderer.toneMappingExposure = parseFloat(e.target.value); }});
elAutoRot.addEventListener('change', e => {{ orbit.autoRotate = e.target.checked; }});
elSpeed.addEventListener('input', e => {{ orbit.autoRotateSpeed = parseFloat(e.target.value); }});

// 변환 모드(W, E, R) 연결
const btnT = document.getElementById('btn-t');
const btnS = document.getElementById('btn-s');
const btnR = document.getElementById('btn-r');

function setTransformMode(mode) {{
    tControl.setMode(mode);
    btnT.className = mode === 'translate' ? 'active' : '';
    btnS.className = mode === 'scale' ? 'active' : '';
    btnR.className = mode === 'rotate' ? 'active' : '';
}}

btnT.addEventListener('click', () => setTransformMode('translate'));
btnS.addEventListener('click', () => setTransformMode('scale'));
btnR.addEventListener('click', () => setTransformMode('rotate'));

window.addEventListener('keydown', e => {{
    if(e.key.toLowerCase() === 'w') setTransformMode('translate');
    if(e.key.toLowerCase() === 'e') setTransformMode('scale');
    if(e.key.toLowerCase() === 'r') setTransformMode('rotate');
}});

// 렌더링 루프
(function animate(){{ 
    requestAnimationFrame(animate); 
    orbit.update(); 
    renderer.render(scene, camera); 
}})();

window.addEventListener('resize', () => {{ 
    renderer.setSize(window.innerWidth, window.innerHeight); 
    camera.aspect = window.innerWidth / window.innerHeight; 
    camera.updateProjectionMatrix(); 
}});
</script>
</body>
</html>"""
st.components.v1.html(html_content, height=800 if not is_single else 650, scrolling=False)