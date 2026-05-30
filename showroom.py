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
    if not os.path.exists(mesh_dir):
        return []
    return sorted(glob.glob(os.path.join(mesh_dir, "*.obj")), reverse=True)


def read_obj_b64(obj_path: str) -> str:
    with open(obj_path, "r", encoding="utf-8") as f:
        content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("utf-8")

params = st.query_params
single_obj_path = params.get("obj", "")

# ---------------------------------------------------------
# 모드 [1] 단일 객체 뷰어 모드
# ---------------------------------------------------------
if single_obj_path:
    single_obj_path = single_obj_path.replace("/", os.sep)
    
    if not os.path.exists(single_obj_path):
        st.error(f"❌ 파일을 찾을 수 없습니다: `{single_obj_path}`")
        st.stop()
        
    st.title("🧊 3D Single Viewer")
    st.caption(f"이 장치는 메인 파일 내부 미리보기(iframe)로 구동됩니다. 파일명: `{os.path.basename(single_obj_path)}`")
    
    obj_b64 = read_obj_b64(single_obj_path)
    
    html_content = f"""<!DOCTYPE html>
    <html lang="ko">
    <head>
      <meta charset="UTF-8"/>
      <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html, body {{ width:100%; height:100%; background:#0d0d14; overflow:hidden; }}
        canvas {{ display:block; width:100%; height:100%; outline:none; }}
        #hint {{ position:fixed; bottom:12px; left:50%; transform:translateX(-50%); color:rgba(180,180,220,0.6); font:12px sans-serif; pointer-events:none; z-index:5; background:rgba(0,0,0,0.5); padding:8px 16px; border-radius:20px; }}
      </style>
    </head>
    <body>
    <canvas id="c"></canvas>
    <div id="hint">좌클릭 드래그: 회전 | 우클릭 드래그: 이동 | 스크롤: 초정밀 줌</div>
    <script type="importmap">{{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }}
    }}</script>
    <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
    
    const objText = new TextDecoder().decode(Uint8Array.from(atob("{obj_b64}"), c=>c.charCodeAt(0)));
    {JS_PARSE_OBJ}
    
    const canvas=document.getElementById('c');
    const renderer=new THREE.WebGLRenderer({{canvas,antialias:true}});
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d0d14);
    const camera=new THREE.PerspectiveCamera(45,window.innerWidth/window.innerHeight,0.01,200);
    camera.position.set(0,0.5,3.5);
    
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.05;
    orbit.autoRotate = true; 
    orbit.autoRotateSpeed = 2.0;
    orbit.minDistance = 1;
    orbit.maxDistance = 30;
    
    // 💡 1. 기본 줌 완전히 끄기
    orbit.enableZoom = false; 
    
    // 💡 2. 수동 줌 이벤트 가로채기 (단일 뷰어)
    canvas.addEventListener('wheel', function(event) {{
        event.preventDefault();
        const ZOOM_SENSITIVITY = 0.0015; // 휠 1틱당 0.15% 스케일 이동
        const distance = camera.position.distanceTo(orbit.target);
        let scale = 1 + (event.deltaY * ZOOM_SENSITIVITY);
        let newDistance = distance * scale;
        newDistance = Math.max(orbit.minDistance, Math.min(orbit.maxDistance, newDistance));
        const direction = new THREE.Vector3().subVectors(camera.position, orbit.target).normalize();
        camera.position.copy(orbit.target).addScaledVector(direction, newDistance);
        orbit.update();
    }}, {{ passive: false }});
    
    scene.add(new THREE.AmbientLight(0xffffff,0.4));
    const dir=new THREE.DirectionalLight(0xffffff,1.2); dir.position.set(5,8,5); scene.add(dir);
    
    const geo=parseOBJ(objText);
    geo.computeBoundingBox();
    const ctr=new THREE.Vector3(); geo.boundingBox.getCenter(ctr); geo.translate(-ctr.x,-ctr.y,-ctr.z);
    const sz=new THREE.Vector3(); geo.boundingBox.getSize(sz); geo.scale(...Array(3).fill(1.8/Math.max(sz.x,sz.y,sz.z)));
    
    const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({{color:0xccccee, metalness:0.15, roughness:0.6, side:THREE.DoubleSide}}));
    scene.add(mesh);
    
    canvas.addEventListener('mousedown', () => orbit.autoRotate = false);
    
    (function animate(){{
      requestAnimationFrame(animate);
      orbit.update();
      renderer.render(scene,camera);
    }})();
    window.addEventListener('resize',()=>{{ renderer.setSize(window.innerWidth,window.innerHeight); camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); }});
    </script>
    </body>
    </html>"""
    st.components.v1.html(html_content, height=650, scrolling=False)

# ---------------------------------------------------------
# 모드 [2] 다중 가상 쇼룸 모드
# ---------------------------------------------------------
else:
    st.title("🏠 3D Virtual Showroom")
    st.caption("생성된 가구 및 객체들을 하나의 3D 가상 쇼룸에 동시 배치하고 자유롭게 인테리어 레이아웃을 조절하세요.")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    os.makedirs(mesh_dir, exist_ok=True)
    
    with st.expander("➕ 외부 3D 모델(.obj) 쇼룸에 직접 추가하기", expanded=False):
        uploaded_obj = st.file_uploader("인터넷에서 다운받거나 직접 만든 .obj 파일을 업로드하면 목록에 바로 추가됩니다.", type=["obj"])
        if uploaded_obj is not None:
            save_path = os.path.join(mesh_dir, uploaded_obj.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_obj.getbuffer())
            st.success(f"✅ `{uploaded_obj.name}` 업로드 성공! 아래 목록에서 체크해주세요.")
    
    st.divider()
    
    obj_files = get_obj_list()
    if not obj_files:
        st.warning("⚠️ 배치할 3D 메쉬 파일이 없습니다. 앱에서 변환하거나 바로 위에서 직접 업로드해 주세요.")
        st.stop()
        
    st.subheader("📦 가상 쇼룸에 배치할 가구 선택")
    selected = []
    cols = st.columns(4)
    for i, f in enumerate(obj_files):
        fname = os.path.basename(f)
        if cols[i % 4].checkbox(fname, key=f"chk_{i}", value=(i < 3)):
            selected.append(f)
            
    if not selected:
        st.info("쇼룸에 렌더링할 오브젝트를 위의 체크박스에서 1개 이상 선택해 주세요.")
        st.stop()
        
    objects_data = []
    for i, path in enumerate(selected):
        objects_data.append({
            "id": i,
            "name": os.path.basename(path),
            "b64": read_obj_b64(path),
            "x": (i % 3) * 2.5 - 2.5,
            "z": (i // 3) * 2.5 - 2.5
        })
        
    objects_json = json.dumps(objects_data)
    
    html_content = f"""<!DOCTYPE html>
    <html lang="ko">
    <head>
      <meta charset="UTF-8"/>
      <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html, body {{ width:100%; height:100%; background:#0d0d14; overflow:hidden; font-family:sans-serif; }}
        canvas {{ display:block; outline: none; }}
        
        #ui {{ position:absolute; top:20px; right:20px; width:220px; background:rgba(30,30,50,0.95); border:1px solid rgba(100,100,200,0.4); border-radius:12px; padding:16px; z-index:20; color:#eee; box-shadow: 0 8px 24px rgba(0,0,0,0.6); backdrop-filter: blur(4px); }}
        #ui h3 {{ color:#bbbbff; font-size:14px; margin-bottom:12px; text-align:center; }}
        .btn-group {{ display: flex; flex-direction: column; gap: 8px; }}
        button {{ background: #2a2a44; color: #fff; border: 1px solid #445; padding: 10px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: 0.2s; }}
        button:hover {{ background: #3a3a5a; border-color:#668; }}
        button.active {{ background: #4a4aff; border-color: #7777ff; font-weight: bold; box-shadow: 0 0 10px rgba(74,74,255,0.4); }}
        
        .color-panel {{ margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center; }}
        .color-panel span {{ font-size: 13px; font-weight: bold; color: #ddd; }}
        #color-picker {{ cursor: pointer; width: 50px; height: 28px; border: none; border-radius: 4px; background: transparent; opacity: 0.5; transition: 0.2s; }}
        #color-picker:not([disabled]) {{ opacity: 1.0; }}
        
        #hint {{ position:fixed; bottom:16px; left:50%; transform:translateX(-50%); color:rgba(220,220,255,0.8); font-size:12px; background:rgba(0,0,0,0.6); padding:10px 20px; border-radius:24px; pointer-events:none; border: 1px solid rgba(255,255,255,0.1); }}
      </style>
    </head>
    <body>
    <canvas id="c"></canvas>
    
    <div id="ui">
        <h3>🖱️ 마우스 컨트롤러</h3>
        <div class="btn-group">
            <button id="btn-t" class="active">✋ 이동 (W)</button>
            <button id="btn-s">🔍 크기 (E)</button>
            <button id="btn-r">🔄 회전 (R)</button>
        </div>
        
        <div class="color-panel">
            <span>🎨 색상 변경</span>
            <input type="color" id="color-picker" value="#ffffff" disabled title="객체를 선택하면 활성화됩니다">
        </div>
        
        <p style="margin-top:16px; font-size:12px; color:#9ab; line-height:1.6; text-align:center;">
            <b>객체 클릭</b> : 선택 및 조작<br>
            <b>허공 클릭</b> : 선택 해제
        </p>
    </div>
    <div id="hint">좌클릭 드래그: 시점 회전 | 우클릭 드래그: 화면 이동 | 스크롤: 초정밀 줌</div>
    
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
    
    const OBJECTS = {objects_json};
    {JS_PARSE_OBJ}
    
    const canvas = document.getElementById('c');
    const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    const scene = new THREE.Scene(); 
    scene.background = new THREE.Color(0x0d0d14); 
    scene.fog = new THREE.Fog(0x0d0d14, 40, 150);
    
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 500);
    camera.position.set(0, 6, 14);
    
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.05;
    orbit.maxPolarAngle = Math.PI / 2 - 0.02; 
    orbit.minDistance = 2;   
    orbit.maxDistance = 60;  
    
    // 💡 1. 꼬이는 기본 줌 기능 완전히 비활성화 (쇼룸 모드)
    orbit.enableZoom = false; 
    
    // 💡 2. 완벽하게 통제되는 물리 엔진 기반 수동 줌 가로채기
    canvas.addEventListener('wheel', function(event) {{
        event.preventDefault(); // 브라우저 스크롤 등 모든 간섭 차단
        
        const ZOOM_SENSITIVITY = 0.0015; // ★ 이 숫자를 수정하여 감도를 마음대로 바꿀 수 있습니다! (0.001 = 더 느려짐)
        
        const distance = camera.position.distanceTo(orbit.target);
        let scale = 1 + (event.deltaY * ZOOM_SENSITIVITY);
        let newDistance = distance * scale;
        
        newDistance = Math.max(orbit.minDistance, Math.min(orbit.maxDistance, newDistance));
        
        const direction = new THREE.Vector3().subVectors(camera.position, orbit.target).normalize();
        camera.position.copy(orbit.target).addScaledVector(direction, newDistance);
        orbit.update();
    }}, {{ passive: false }});
    
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 1.5); 
    dir.position.set(10, 15, 10); 
    dir.castShadow = true; 
    dir.shadow.mapSize.width = 2048; 
    dir.shadow.mapSize.height = 2048;
    scene.add(dir);
    
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(80,80), new THREE.MeshStandardMaterial({{color:0x151525, roughness:0.8}}));
    floor.rotation.x = -Math.PI/2; 
    floor.position.y = -1; 
    floor.receiveShadow = true; 
    scene.add(floor);
    
    const grid = new THREE.GridHelper(80, 80, 0x333355, 0x1a1a33); 
    grid.position.y = -0.99; 
    scene.add(grid);
    
    const COLORS = [0x6688ff, 0xff8866, 0x66ff88, 0xffcc44, 0xff66aa];
    const interactableMeshes = [];
    
    for (const obj of OBJECTS) {{
      const geo = parseOBJ(new TextDecoder().decode(Uint8Array.from(atob(obj.b64), c=>c.charCodeAt(0))));
      geo.computeBoundingBox();
      const ctr = new THREE.Vector3(); geo.boundingBox.getCenter(ctr); geo.translate(-ctr.x, -ctr.y, -ctr.z);
      const sz = new THREE.Vector3(); geo.boundingBox.getSize(sz); geo.scale(...Array(3).fill(2.0/Math.max(sz.x, sz.y, sz.z)));
      
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({{color:COLORS[obj.id%COLORS.length], roughness:0.5, metalness:0.1, side:THREE.DoubleSide}}));
      mesh.castShadow = true;
      mesh.receiveShadow = true; 
      mesh.position.set(obj.x, 0, obj.z);
      scene.add(mesh); 
      interactableMeshes.push(mesh);
    }}
    
    const transformControl = new TransformControls(camera, renderer.domElement);
    transformControl.addEventListener('dragging-changed', function (event) {{
        orbit.enabled = !event.value;
    }});
    scene.add(transformControl);
    
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let selectedObject = null;
    
    const colorPicker = document.getElementById('color-picker');
    colorPicker.addEventListener('input', (event) => {{
        if (selectedObject && selectedObject.material) {{
            selectedObject.material.color.set(event.target.value);
        }}
    }});
    
    canvas.addEventListener('pointerdown', function(event) {{
        if (transformControl.dragging) return; 
        
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(interactableMeshes, true);
        
        if (intersects.length > 0) {{
            selectedObject = intersects[0].object;
            transformControl.attach(selectedObject);
            
            colorPicker.disabled = false;
            colorPicker.value = '#' + selectedObject.material.color.getHexString();
        }} else {{
            transformControl.detach();
            selectedObject = null;
            
            colorPicker.disabled = true;
            colorPicker.value = '#ffffff';
        }}
    }});
    
    const btnT = document.getElementById('btn-t');
    const btnS = document.getElementById('btn-s');
    const btnR = document.getElementById('btn-r');
    
    function setMode(mode) {{
        transformControl.setMode(mode);
        btnT.className = mode === 'translate' ? 'active' : '';
        btnS.className = mode === 'scale' ? 'active' : '';
        btnR.className = mode === 'rotate' ? 'active' : '';
    }}
    
    btnT.addEventListener('click', () => setMode('translate'));
    btnS.addEventListener('click', () => setMode('scale'));
    btnR.addEventListener('click', () => setMode('rotate'));
    
    window.addEventListener('keydown', function(event) {{
        switch (event.key.toLowerCase()) {{
            case 'w': setMode('translate'); break; 
            case 'e': setMode('scale'); break;     
            case 'r': setMode('rotate'); break;    
        }}
    }});
    
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
    st.components.v1.html(html_content, height=800, scrolling=False)