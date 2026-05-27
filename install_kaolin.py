import torch
import subprocess
import sys

# 현재 설치된 PyTorch 버전을 감지합니다 (예: 2.4.0)
torch_version = torch.__version__.split('+')[0]
url = f"https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{torch_version}_cu121.html"

print("==================================================")
print(f"🔍 감지된 PyTorch 버전: {torch_version}")
print(f"🔗 Kaolin 다운로드 타겟: {url}")
print("==================================================")

# 1. Kaolin 설치 (사전 빌드된 윈도우 파일)
print("\n🚀 [1/2] NVIDIA Kaolin 설치를 시작합니다...")
subprocess.run([sys.executable, "-m", "pip", "install", "kaolin", "-f", url])

# 2. Diso 설치
print("\n🚀 [2/2] Diso 라이브러리 설치를 시작합니다...")
subprocess.run([sys.executable, "-m", "pip", "install", "diso"])

print("\n✅ 모든 3D 필수 라이브러리 설치가 완료되었습니다!")