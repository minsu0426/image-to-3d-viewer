import os
import sys
import subprocess
import time
import shutil
from PIL import Image

def run_modeb_instantmesh(images, output_dir=None):
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(pipeline_dir)
    target_output_dir = os.path.join(project_root, "outputs", "meshes")
    os.makedirs(target_output_dir, exist_ok=True)

    core_dir = os.path.join(pipeline_dir, "instantmesh_core")
    
    if not os.path.exists(core_dir):
        raise FileNotFoundError("InstantMesh 코어 엔진이 없습니다. 터미널에서 git clone 설치를 진행해주세요.")
        
    # 💡 범용성 1: 캔버스 크기를 512로 고정 (VRAM이 버틴다면 디테일 상승)
    best_frame = images[0]
    target_canvas_size = 512
    
    # 💡 범용성 2: 투명 배경의 빈 공간(노이즈)을 타이트하게 잘라냅니다.
    if best_frame.mode == 'RGBA':
        bbox = best_frame.getbbox() 
        if bbox:
            best_frame = best_frame.crop(bbox)
            
    # 💡 범용성 3: '적응형 스케일링' (오버피팅 방지)
    # 물체가 길쭉하든 납작하든, 가장 긴 쪽을 캔버스의 85%에 맞추어 원본 비율을 철저히 유지합니다.
    safe_margin = 0.85
    max_dim = max(best_frame.width, best_frame.height)
    scale_factor = (target_canvas_size * safe_margin) / max_dim
    
    new_w = int(best_frame.width * scale_factor)
    new_h = int(best_frame.height * scale_factor)
    best_frame = best_frame.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 하얀색 도화지를 깔고, 어떤 모양이든 무조건 정중앙에 오도록 좌표 계산
    white_bg = Image.new("RGB", (target_canvas_size, target_canvas_size), (255, 255, 255))
    offset_x = (target_canvas_size - new_w) // 2
    offset_y = (target_canvas_size - new_h) // 2
    
    if best_frame.mode == 'RGBA' and len(best_frame.split()) == 4:
        white_bg.paste(best_frame, (offset_x, offset_y), mask=best_frame.split()[3])
    else:
        white_bg.paste(best_frame, (offset_x, offset_y))
        
    best_frame = white_bg
    
    temp_input_dir = os.path.join(pipeline_dir, "temp_input")
    temp_output_dir = os.path.join(pipeline_dir, "temp_output")
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    
    input_filename = f"target_{int(time.time())}.png"
    input_filepath = os.path.join(temp_input_dir, input_filename)
    best_frame.save(input_filepath, "PNG")
    
    command = [
        sys.executable, "run.py",
        "configs/instant-mesh-base.yaml",
        input_filepath,
        "--output_path", temp_output_dir,
        "--no_rembg",
        "--diffusion_steps", "75" # 범용성을 위해 75로 롤백 (100은 일부 모델에서 타는 현상 유발)
    ]
    
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    print("\n==============================================")
    print("🚀 [진행 중] InstantMesh 3D 엔진 가동 시작!")
    print(f"📂 [경로 안내] 최종 메쉬 저장 위치: {target_output_dir}")
    print("==============================================\n")
    
    try:
        subprocess.run(command, cwd=core_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_input_dir): shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
        raise RuntimeError("엔진 실행 중 에러가 발생했습니다. 터미널 로그를 확인하세요.")
        
    obj_files = []
    for root, dirs, files in os.walk(temp_output_dir):
        for f in files:
            if f.endswith('.obj'):
                obj_files.append(os.path.join(root, f))
    
    if not obj_files:
        if os.path.exists(temp_input_dir): shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
        raise FileNotFoundError(f"3D 메쉬(.obj) 파일이 생성되지 않았습니다.")
        
    result_file_path = obj_files[0]
    final_dest = os.path.join(target_output_dir, f"modeB_LRM_{int(time.time())}.obj")
    shutil.move(result_file_path, final_dest)
    
    shutil.rmtree(temp_input_dir)
    shutil.rmtree(temp_output_dir)
    
    print(f"\n✅ [성공] 3D 메쉬 파일이 안전하게 저장되었습니다: {final_dest}\n")
    
    return final_dest