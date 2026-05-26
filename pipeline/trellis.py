import os
import time
import torch
import gc
import streamlit as st
import trimesh
from PIL import Image

@st.cache_resource
def load_trellis_model():
    """TRELLIS 파이프라인 가중치를 VRAM에 로드 (캐싱)"""
    from trellis.pipelines import TrellisImageTo3DPipeline
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TRELLIS Large 모델 로드 (약 3~4GB 소요)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    return pipeline, device

def run_modea_trellis(image: Image.Image, output_dir: str, resolution: int = 256) -> str:
    """단일 이미지 -> TRELLIS -> 3D 메쉬(.obj) 추출"""
    pipeline, device = load_trellis_model()
    
    # 이미 SAM 2로 누끼를 땄으므로, TRELLIS 자체 배경 제거(rembg)는 False로 설정해 속도를 높입니다.
    with torch.no_grad():
        outputs = pipeline.run(
            image, 
            formats=["mesh"], 
            preprocess_image=False  # SAM2 마스크 그대로 사용!
        )
        
    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 메쉬 데이터 추출
    mesh_data = outputs['mesh'][0]
    
    # 파일 저장 경로 설정
    timestamp = int(time.time())
    obj_path = os.path.normpath(os.path.join(output_dir, f"mesh_modeA_TRELLIS_{timestamp}.obj"))
    
    # TRELLIS가 추출한 텐서 데이터를 표준 trimesh 객체로 변환하여 저장
    if hasattr(mesh_data, 'export'):
        mesh_data.export(obj_path)
    else:
        # Raw 데이터일 경우 직접 Trimesh 포맷으로 래핑
        vertices = mesh_data.vertices.detach().cpu().numpy()
        faces = mesh_data.faces.detach().cpu().numpy()
        t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        t_mesh.export(obj_path)
        
    return obj_path