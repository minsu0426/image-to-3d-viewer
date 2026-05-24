import streamlit as st
import streamlit.components.v1 as components
import os
import base64

st.set_page_config(page_title="3D Viewer", page_icon="🧊", layout="wide")

# URL 파라미터에서 .obj 경로 수신
params = st.query_params
obj_path = params.get("obj", "")

# Windows 경로 슬래시 정규화
obj_path = obj_path.replace("/", os.sep)

st.title("🧊 3D Viewer")

if not obj_path:
    st.warning("⚠️ .obj 파일 경로가 전달되지 않았습니다. app.py에서 '3D 뷰어 열기' 버튼을 클릭하세요.")
    st.stop()

if not os.path.exists(obj_path):
    st.error(f"❌ 파일을 찾을 수 없습니다: `{obj_path}`")
    st.stop()

st.caption(f"파일: `{obj_path}`")

# .obj → Base64 인코딩
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
    canvas {{ display:block; width:100%; height:100%; }}
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
  const uvArr =uvs.length   ?new Float32Array(verts.length*2):null;
  verts.forEach((v,i)=>{{
    posArr[i*3]  =positions[v.vi*3];
    posArr[i*3+1]=positions[v.vi*3+1];
    posArr[i*3+2]=positions[v.vi*3+2];
    if(nrmArr&&v.ni!==undefined){{
      nrmArr[i*3]=normals[v.ni*3];
      nrmArr[i*3+1]=normals[v.ni*3+1];
      nrmArr[i*3+2]=normals[v.ni*3+2];
    }}
    if(uvArr&&v.ti!==undefined){{
      uvArr[i*2]  =uvs[v.ti*2];
      uvArr[i*2+1]=uvs[v.ti*2+1];
    }}
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
renderer.setSize(window.innerWidth, window.innerHeight);
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

const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({{
  color:0xccccee, metalness:0.15, roughness:0.6, side:THREE.DoubleSide
}}));
mesh.castShadow=mesh.receiveShadow=true;
scene.add(mesh);

const grid=new THREE.GridHelper(4,20,0x333366,0x222244);
grid.position.y=-1; scene.add(grid);

document.getElementById('loading').style.opacity='0';
setTimeout(()=>document.getElementById('loading').remove(),500);

let drag=false,rDrag=false,px=0,py=0;
let rotX=0,rotY=0,panX=0,panY=0,zoom=2.5,auto=true;

canvas.addEventListener('mousedown',e=>{{
  drag=true; rDrag=e.button===2;
  px=e.clientX; py=e.clientY; auto=false;
}});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{{
  if(!drag)return;
  const dx=e.clientX-px, dy=e.clientY-py;
  px=e.clientX; py=e.clientY;
  if(rDrag){{ panX+=dx*0.004; panY-=dy*0.004; }}
  else     {{ rotY+=dx*0.6;   rotX+=dy*0.6; }}
}});
canvas.addEventListener('wheel',e=>{{
  e.preventDefault();
  zoom=Math.max(0.5,Math.min(10,zoom+e.deltaY*0.005));
}},{{passive:false}});
canvas.addEventListener('contextmenu',e=>e.preventDefault());

const clock=new THREE.Clock();
(function animate(){{
  requestAnimationFrame(animate);
  const dt=clock.getDelta();
  if(auto) rotY+=dt*25;
  mesh.rotation.set(
    THREE.MathUtils.degToRad(rotX),
    THREE.MathUtils.degToRad(rotY), 0
  );
  camera.position.set(panX, 0.5+panY, zoom);
  camera.lookAt(panX, panY, 0);
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

components.html(html, height=800, scrolling=False)