import os
import sys
import subprocess
import time
import shutil
from PIL import Image

def run_modeb_instantmesh(images, output_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.join(base_dir, "instantmesh_core")
    
    if not os.path.exists(core_dir):
        raise FileNotFoundError("InstantMesh 코어 엔진이 없습니다. 터미널에서 git clone 설치를 진행해주세요.")
        
    best_frame = images[0]
    
    temp_input_dir = os.path.join(base_dir, "temp_input")
    temp_output_dir = os.path.join(base_dir, "temp_output")
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    
    input_filename = f"target_{int(time.time())}.png"
    input_filepath = os.path.join(temp_input_dir, input_filename)
    
    best_frame.save(input_filepath, "PNG")
    
    # 💡 수정 1: "python" 대신 sys.executable을 써서 무조건 현재 venv 가상환경의 파이썬을 사용하도록 강제합니다!
    command = [
        sys.executable, "run.py",
        "configs/instant-mesh-large.yaml",
        input_filepath,
        "--output_path", temp_output_dir
    ]
    
    try:
        # 💡 수정 2: capture_output=True, text=True 를 추가하여 숨겨진 에러 로그를 낚아챕니다.
        subprocess.run(command, cwd=core_dir, capture_output=True, text=True, check=True)
        
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_input_dir): shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
        # 💡 수정 3: 진짜 에러 내용(e.stderr)을 Streamlit 화면에 빨간 글씨로 크게 띄워줍니다!
        error_msg = f"🚨 InstantMesh 엔진 에러 상세 기록:\n\n{e.stderr}"
        raise RuntimeError(error_msg)
        
    generated_mesh_dir = os.path.join(temp_output_dir, "meshes")
    
    if not os.path.exists(generated_mesh_dir):
        if os.path.exists(temp_input_dir): shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
        raise FileNotFoundError("InstantMesh가 meshes 폴더를 생성하지 못했습니다. (VRAM 부족이나 라이브러리 호환성 문제일 수 있습니다)")
        
    obj_files = [f for f in os.listdir(generated_mesh_dir) if f.endswith('.obj')]
    
    if not obj_files:
        raise FileNotFoundError("3D 메쉬(.obj) 파일이 생성되지 않았습니다.")
        
    result_file = obj_files[0]
    final_dest = os.path.join(output_dir, f"modeB_LRM_{int(time.time())}.obj")
    shutil.move(os.path.join(generated_mesh_dir, result_file), final_dest)
    
    shutil.rmtree(temp_input_dir)
    shutil.rmtree(temp_output_dir)
    
    return final_dest