# 🚀 Image-to-3D Viewer & Virtual Showroom

SAM 2(Segment Anything 2)와 TRELLIS, 그리고 InstantMesh를 결합한 지능형 2D-to-3D 변환 및 가상 쇼룸 시스템입니다. 복잡한 3D 모델링 지식 없이도 사진 한 장으로 정교한 3D 메쉬를 생성하고, 이를 브라우저 기반의 가상 공간에서 직접 배치하고 시각화할 수 있습니다.

## 🌟 1. 핵심 기능 (Core Features)

* **정밀 객체 세그멘테이션 (SAM 2 Integration):** 사용자의 이미지를 분석하여 텍스트 파편화나 배경 노이즈 없이 객체만 픽셀 단위로 정교하게 분리하고 투명 배경(RGBA)으로 추출합니다. 추출 후 OpenCV를 통한 정밀 노이즈 제거 및 테두리 평탄화(Erosion & Blur) 작업이 적용됩니다.
* **고품질 3D 메쉬 생성 (TRELLIS & InstantMesh Engine):** 추출된 2D 이미지를 입력받아 최신 3D 파운데이션 모델을 통해 다각도를 상상하고, 기하학적 구조가 정교한 3D 메쉬(.obj)를 생성합니다. 생성 후 Open3D를 이용한 표면 다림질(Taubin Smoothing) 알고리즘이 적용됩니다.
* **인터랙티브 3D 가상 쇼룸 (3D Visualization):** Three.js 기반의 독립된 렌더링 엔진(`showroom.py`)을 통해 생성된 3D 객체들을 격자 형태의 쇼룸에 동시 배치합니다. 마우스 드래그를 통한 자유로운 이동/크기/회전 조작과 실시간 색상(Color) 변경을 지원합니다.

---

## 🔄 2. 시스템 파이프라인 (System Pipeline)

시스템은 `Port 8501`(메인 AI 추론)과 `Port 8502`(3D 웹 렌더링)로 완전히 분리된 마이크로서비스 아키텍처로 동작합니다.

1. **Input Phase:** 웹 UI(`app.py`)를 통해 단일 2D 이미지를 업로드합니다.
2. **Segmentation Phase:** SAM 2가 객체 외곽을 분석하여 알파 채널(투명도) 마스크가 적용된 이미지를 추출합니다.
3. **Reconstruction Phase:** 추출된 이미지가 3D 파이프라인으로 전달되어 3D 포인트 클라우드 생성 ➔ FlexiCubes & Kaolin 엔진을 통한 메쉬(.obj) 추출을 수행합니다.
4. **Rendering Phase:** 생성된 메쉬 파일은 로컬에 아카이빙되며, `showroom.py` (Three.js) 엔진을 통해 인라인 뷰어 또는 다중 객체 쇼룸 형태로 브라우저에 실시간 렌더링됩니다.

---

## 🛠️ 3. 설치 및 실행 가이드 (Getting Started)

본 프로젝트는 고성능 3D 렌더링 연산을 위해 **NVIDIA GPU (CUDA)** 및 **Python 3.10** 환경이 필수적으로 요구됩니다. 컴파일러 충돌 및 코어 엔진 누락을 막기 위해 아래의 설치 가이드를 엄격히 따라주세요.

### 📋 시스템 요구사항 및 사전 준비 (Prerequisites)
* **OS:** Windows 10/11 또는 Linux
* **Python:** `3.10.x` (3.11 이상 버전은 3D 컴파일러와 호환되지 않으므로 엄격히 제한)
* **Hardware:** NVIDIA GPU (VRAM 8GB 이상, RTX 3060 등 권장), RAM 16GB 이상
* **Compiler (Windows 필수):** 1. `NVIDIA CUDA Toolkit 12.1` (설치 시 Visual Studio Integration 체크 해제 권장)
  2. `Visual Studio 2022 C++ Build Tools` (14.38 버전 권장. 설치 시 cl.exe 경로 인식을 위해 파워쉘 대신 **'VS 2022용 x64 Native Tools 명령 프롬프트'** 사용을 강력히 권장합니다.)

### ⚙️ 환경 세팅 및 라이브러리 설치

**1. Python 3.10 가상환경 생성 및 활성화**
```cmd
python -m venv venv

# Windows (일반 cmd 또는 x64 Native Tools cmd 창에서 실행)
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate