import streamlit as st
import streamlit.components.v1 as components
import os
import json
import base64
import glob

st.set_page_config(page_title="3D Showroom & Viewer", page_icon="🏠", layout="wide")

# =========================================================
# 공통 유틸리티 함수
# =========================================================

def get_obj_list() -> list:
    """쇼룸 모드에서 사용할 전체 .obj 파일 목록 로드"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.normpath(os.path.join(base_dir, "outputs", "meshes"))
    if not os.path.exists(mesh_dir):
        return []
    return sorted(glob.glob(os.path.join(mesh_dir, "*.obj")), reverse=True)


def read_obj_b64(obj_path: str) -> str:
    """Three.js 내부로 안전하게 주입하기 위한 Base64 변환"""
    with open(obj_path, "r", encoding="utf-8") as f:
        content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("utf-8")


# =========================================================
# URL 파라미터 분석을 통한 모드 분기
# =========================================================
params = st.query_params
single_obj_path = params.get("obj", "")

# ---------------------------------------------------------
# 모드 [1] 단일 객체 뷰어 모드 (?obj=경로 가 주어졌을 때)
# ---------------------------------------------------------
if single_obj_path:
    single_obj_path = single_obj_path.replace("/", os.sep)
    
    if not os.path.exists(single_obj_path):
        st.error(f"❌ 파일을 찾을 수 없습니다: `{single_obj_path}`")
        st.stop()
        
    st.title("🧊 3D Single Viewer")
    st.caption(f"이 장치는 메인 파일 내부 미리보기(iframe)로 구동됩니다. 파일명: `{os.path.basename(single_obj_path)}`")
    
    obj_b64 = read_obj_b64(single_obj_path)
    
    # 단일 뷰어 전용 HTML (자동 회전 기능 포함, 컨트롤 가벼움)
    html_content = f"""<!DOCTYPE html>
    <html lang="ko">
    <head>
      <meta charset="UTF-8"/>
      <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html, body {{ width:100%; height:100%; background:#0d0d14; overflow:hidden; }}
        canvas {{ display:block; width:100%; height:100%; }}
        #hint {{ position:fixed; bottom:12px; left:50%; transform:translateX(-50%); color:rgba(180,180,220,0.6); font:11px sans-serif; pointer-events:none; z-index:5; }}
      </style>
    </head>
    <body>
    <canvas id="c"></canvas>
    <div id="hint">🖱 드래그: 회전 | 스크롤: 줌 | 우클릭 드래그: 이동</div>
    <script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}}}</script>
    <script type="module">
    import * as THREE from 'three';
    
    const objText = new TextDecoder().decode(Uint8Array.from(atob("{obj_b64}"), c=>c.charCodeAt(0)));
    {"" if False else "function parseOBJ(text) { const positions=[], normals=[], uvs=[], verts=[]; for (const raw of text.split('\\n')) { const line=raw.trim(); if (!line || line.startsWith('#')) continue; const p=line.split(/\\s+/); if (p[0]==='v') positions.push(+p[1],+p[2],+p[3]); else if (p[0]==='vn') normals.push(+p[1],+p[2],+p[3]); else if (p[0]==='f') { const fv=p.slice(1); for (let i=1;i<fv.length-1;i++) [fv[0],fv[i],fv[i+1]].forEach(t=>{ const [vi,ti,ni]=t.split('/').map(x=>x?+x-1:undefined); verts.push({vi,ti,ni}); }); } } const posArr=new Float32Array(verts.length*3); verts.forEach((v,i)=>{ posArr[i*3]=positions[v.vi*3]; posArr[i*3+1]=positions[v.vi*3+1]; posArr[i*3+2]=positions[v.vi*3+2]; }); const geo=new THREE.BufferGeometry(); geo.setAttribute('position',new THREE.BufferAttribute(posArr,3)); geo.computeVertexNormals(); return geo; }"}
    
    const canvas=document.getElementById('c');
    const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d0d14);
    const camera=new THREE.PerspectiveCamera(45,window.innerWidth/window.innerHeight,0.01,100);
    camera.position.set(0,0.5,2.5);
    
    scene.add(new THREE.AmbientLight(0xffffff,0.4));
    const dir=new THREE.DirectionalLight(0xffffff,1.2); dir.position.set(5,8,5); scene.add(dir);
    
    const geo=parseOBJ(objText);
    geo.computeBoundingBox();
    const ctr=new THREE.Vector3(); geo.boundingBox.getCenter(ctr); geo.translate(-ctr.x,-ctr.y,-ctr.z);
    const sz=new THREE.Vector3(); geo.boundingBox.getSize(sz); geo.scale(...Array(3).fill(1.8/Math.max(sz.x,sz.y,sz.z)));
    
    const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0xccccee, metalness:0.15, roughness:0.6, side:THREE.DoubleSide}));
    scene.add(mesh);
    
    let drag=false,rDrag=false,px=0,py=0;
    let rotX=0,rotY=0,zoom=2.5,auto=true;
    
    canvas.addEventListener('mousedown',e=>{{ drag=true; rDrag=e.button===2; px=e.clientX; py=e.clientY; auto=false; }});
    window.addEventListener('mousemove',e=>{{
      if(!drag)return; const dx=e.clientX-px, dy=e.clientY-py; px=e.clientX; py=e.clientY;
      if(!rDrag) {{ rotY+=dx*0.6; rotX+=dy*0.6; }}
    }});
    window.addEventListener('mouseup',()=>drag=false);
    canvas.addEventListener('wheel',e=>{{ e.preventDefault(); zoom=Math.max(0.5,Math.min(10,zoom+e.deltaY*0.005)); }},{{passive:false}});
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    
    const clock=new THREE.Clock();
    (function animate(){{
      requestAnimationFrame(animate);
      if(auto) rotY+=clock.getDelta()*25;
      mesh.rotation.set(THREE.MathUtils.degToRad(rotX), THREE.MathUtils.degToRad(rotY), 0);
      camera.position.set(0, 0.5, zoom); camera.lookAt(0, 0, 0);
      renderer.render(scene,camera);
    }})();
    window.addEventListener('resize',()=>{{ renderer.setSize(window.innerWidth,window.innerHeight); camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); }});
    </script>
    </body>
    </html>"""
    st.iframe(html_content, height=650, scrolling=False)
# ---------------------------------------------------------
# 모드 [2] 다중 가상 쇼룸 모드 (기본 파라미터가 없을 때)
# ---------------------------------------------------------
else:
    st.title("🏠 3D Virtual Showroom")
    st.caption("생성된 가구 및 객체들을 하나의 3D 가상 쇼룸에 동시 배치하고 자유롭게 인테리어 레이아웃을 조절하세요.")
    
    obj_files = get_obj_list()
    if not obj_files:
        st.warning("⚠️ outputs/meshes/ 폴더에 생성된 3D 메쉬 파일이 존재하지 않습니다.")
        st.stop()
        
    st.subheader("📦 가상 쇼룸에 배치할 가구 선택")
    selected = []
    cols = st.columns(4)
    for i, f in enumerate(obj_files):
        fname = os.path.basename(f)
        if cols[i % 4].checkbox(fname, key=f"chk_{i}", value=(i < 3)):
            selected.append(f)
            
    if not selected:
        st.info("쇼룸에 렌더링할 오브젝트를 좌측 체크박스에서 1개 이상 선택해 주세요.")
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
    
    # 다중 쇼룸 전용 HTML (슬라이더 컨트롤러 활성화 및 격자 배치형)
    html_content = f"""<!DOCTYPE html>
    <html lang="ko">
    <head>
      <meta charset="UTF-8"/>
      <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html, body {{ width:100%; height:100%; background:#0d0d14; overflow:hidden; font-family:sans-serif; }}
        canvas {{ display:block; }}
        #ui {{ position:fixed; top:12px; right:12px; width:240px; background:rgba(20,20,40,0.92); border:1px solid rgba(100,100,200,0.3); border-radius:12px; padding:14px; z-index:20; color:#ccc; font-size:12px; max-height:calc(100vh - 24px); overflow-y:auto; }}
        #ui h3 {{ color:#aaaaff; font-size:13px; margin-bottom:10px; }}
        .obj-item {{ background:rgba(60,60,100,0.3); border-radius:8px; padding:8px; margin-bottom:8px; }}
        .obj-item .name {{ font-weight:bold; color:#ddd; font-size:11px; margin-bottom:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .slider-row {{ display:flex; align-items:center; gap:6px; margin-top:4px; }}
        .slider-row label {{ width:44px; color:#999; font-size:10px; flex-shrink:0; }}
        .slider-row input[type=range] {{ flex:1; height:3px; accent-color:#6666ff; }}
        .slider-row span {{ width:32px; text-align:right; color:#aaa; font-size:10px; }}
        #hint {{ position:fixed; bottom:12px; left:50%; transform:translateX(-50%); color:rgba(180,180,220,0.5); font-size:11px; pointer-events:none; }}
      </style>
    </head>
    <body>
    <canvas id="c"></canvas>
    <div id="ui"><h3>🎛️ 오브젝트 컨트롤</h3><div id="obj-list"></div></div>
    <div id="hint">🖱 드래그: 카메라 회전 | 스크롤: 줌 | 우클릭 드래그: 시점 이동</div>
    
    <script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}}}</script>
    <script type="module">
    import * as THREE from 'three';
    
    const OBJECTS = {objects_json};
    {"" if False else "function parseOBJ(text) { const positions=[], normals=[], uvs=[], verts=[]; for (const raw of text.split('\\n')) { const line=raw.trim(); if (!line || line.startsWith('#')) continue; const p=line.split(/\\s+/); if (p[0]==='v') positions.push(+p[1],+p[2],+p[3]); else if (p[0]==='vn') normals.push(+p[1],+p[2],+p[3]); else if (p[0]==='f') { const fv=p.slice(1); for (let i=1;i<fv.length-1;i++) [fv[0],fv[i],fv[i+1]].forEach(t=>{ const [vi,ti,ni]=t.split('/').map(x=>x?+x-1:undefined); verts.push({vi,ti,ni}); }); } } const posArr=new Float32Array(verts.length*3); verts.forEach((v,i)=>{ posArr[i*3]=positions[v.vi*3]; posArr[i*3+1]=positions[v.vi*3+1]; posArr[i*3+2]=positions[v.vi*3+2]; }); const geo=new THREE.BufferGeometry(); geo.setAttribute('position',new THREE.BufferAttribute(posArr,3)); geo.computeVertexNormals(); return geo; }"}
    
    const canvas=document.getElementById('c');
    const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
    renderer.setSize(window.innerWidth,window.innerHeight);
    renderer.shadowMap.enabled=true;
    
    const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d0d14); scene.fog=new THREE.Fog(0x0d0d14,20,60);
    const camera=new THREE.PerspectiveCamera(50,window.innerWidth/window.innerHeight,0.01,200);
    camera.position.set(0,4,10);
    
    scene.add(new THREE.AmbientLight(0xffffff,0.5));
    const dir=new THREE.DirectionalLight(0xffffff,1.2); dir.position.set(8,12,8); dir.castShadow=true; scene.add(dir);
    
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(40,40), new THREE.MeshStandardMaterial({color:0x111122,roughness:0.9}));
    floor.rotation.x=-Math.PI/2; floor.position.y=-1; floor.receiveShadow=true; scene.add(floor);
    const grid=new THREE.GridHelper(40,40,0x222244,0x1a1a33); grid.position.y=-0.99; scene.add(grid);
    
    const COLORS=[0x6688ff,0xff8866,0x66ff88,0xffcc44,0xff66aa];
    const meshes={};
    
    for (const obj of OBJECTS) {{
      const geo=parseOBJ(new TextDecoder().decode(Uint8Array.from(atob(obj.b64),c=>c.charCodeAt(0))));
      geo.computeBoundingBox();
      const ctr=new THREE.Vector3(); geo.boundingBox.getCenter(ctr); geo.translate(-ctr.x,-ctr.y,-ctr.z);
      const sz=new THREE.Vector3(); geo.boundingBox.getSize(sz); geo.scale(...Array(3).fill(1.5/Math.max(sz.x,sz.y,sz.z)));
      
      const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:COLORS[obj.id%COLORS.length], roughness:0.6, side:THREE.DoubleSide}));
      mesh.castShadow=mesh.receiveShadow=true; mesh.position.set(obj.x, 0, obj.z);
      scene.add(mesh); meshes[obj.id]=mesh;
    }}
    
    const listEl=document.getElementById('obj-list');
    for (const obj of OBJECTS) {{
      const div=document.createElement('div'); div.className='obj-item';
      div.innerHTML=`
        <div class="name">${{obj.name}}</div>
        <div class="slider-row"><label>크기</label><input type="range" id="scale-${{obj.id}}" min="0.1" max="4" step="0.05" value="1.0"><span id="scale-val-${{obj.id}}">1.0</span></div>
        <div class="slider-row"><label>X 위치</label><input type="range" id="px-${{obj.id}}" min="-8" max="8" step="0.1" value="${{obj.x}}"><span id="px-val-${{obj.id}}">${{obj.x}}</span></div>
        <div class="slider-row"><label>Z 위치</label><input type="range" id="pz-${{obj.id}}" min="-8" max="8" step="0.1" value="${{obj.z}}"><span id="pz-val-${{obj.id}}">${{obj.z}}</span></div>
        <div class="slider-row"><label>Y 회전</label><input type="range" id="ry-${{obj.id}}" min="0" max="360" step="1" value="0"><span id="ry-val-${{obj.id}}">0°</span></div>
      `;
      listEl.appendChild(div);
      
      document.getElementById(`scale-${{obj.id}}`).addEventListener('input', e=>{{ meshes[obj.id].scale.setScalar(parseFloat(e.target.value)); document.getElementById(`scale-val-${{obj.id}}`).textContent=e.target.value; }});
      document.getElementById(`px-${{obj.id}}`).addEventListener('input', e=>{{ meshes[obj.id].position.x=parseFloat(e.target.value); document.getElementById(`px-val-${{obj.id}}`).textContent=e.target.value; }});
      document.getElementById(`pz-${{obj.id}}`).addEventListener('input', e=>{{ meshes[obj.id].position.z=parseFloat(e.target.value); document.getElementById(`pz-val-${{obj.id}}`).textContent=e.target.value; }});
      document.getElementById(`ry-${{obj.id}}`).addEventListener('input', e=>{{ meshes[obj.id].rotation.y=THREE.MathUtils.degToRad(parseFloat(e.target.value)); document.getElementById(`ry-val-${{obj.id}}`).textContent=e.target.value+'°'; }});
    }}
    
    let drag=false, rDrag=false, px=0, py=0, theta=0, phi=30, radius=12, panX=0, panZ=0;
    function updateCamera() {{
      const t=THREE.MathUtils.degToRad(theta), p=THREE.MathUtils.degToRad(phi);
      camera.position.set(panX + radius*Math.sin(t)*Math.cos(p), radius*Math.sin(p), panZ + radius*Math.cos(t)*Math.cos(p));
      camera.lookAt(panX, 0, panZ);
    }}
    updateCamera();
    
    canvas.addEventListener('mousedown',e=>{{ drag=true; rDrag=e.button===2; px=e.clientX; py=e.clientY; }});
    window.addEventListener('mousemove',e=>{{
      if(!drag)return; const dx=e.clientX-px, dy=e.clientY-py; px=e.clientX; py=e.clientY;
      if(rDrag) {{ panX-=dx*0.02; panZ+=dy*0.02; }} else {{ theta-=dx*0.4; phi=Math.max(5,Math.min(85,phi-dy*0.4)); }} updateCamera();
    }});
    window.addEventListener('mouseup',()=>drag=false);
    canvas.addEventListener('wheel',e=>{{ e.preventDefault(); radius=Math.max(2,Math.min(30,radius+e.deltaY*0.02)); updateCamera(); }},{{passive:false}});
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    
    (function animate(){{ requestAnimationFrame(animate); renderer.render(scene,camera); }})();
    window.addEventListener('resize',()=>{{ renderer.setSize(window.innerWidth,window.innerHeight); camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); }});
    </script>
    </body>
    </html>"""
    st.iframe(html_content, height=750, scrolling=False)