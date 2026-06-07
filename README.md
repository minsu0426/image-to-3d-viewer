# 🚀 3D Studio Pro: Image-to-3D Viewer & Virtual Showroom

**3D Studio Pro**는 단일 이미지 또는 다중 프레임(GIF/동영상)을 입력받아 고품질 3D 메쉬(.obj)로 복원하고, 이를 브라우저 기반의 3D 가상 공간(Showroom)에서 실시간으로 편집 및 렌더링할 수 있는 **지능형 2D-to-3D 파이프라인 시스템**입니다.

SAM 2(Segment Anything 2), TRELLIS, InstantMesh 등 최신 AI 파운데이션 모델을 마이크로서비스 형태로 결합하여 복잡한 3D 모델링 지식 없이도 누구나 정교한 3D 에셋을 생성할 수 있습니다.

---

## 🌟 1. 핵심 기능 및 기술 스택

### 🔹 1-1. 정밀 객체 세그멘테이션 (Hybrid AI Segmentation)

- **Tech:** `Rembg (U2Net)` + `SAM 2 (Segment Anything 2)` + `OpenCV`
- **Feature:** Rembg로 1차 Bounding Box를 추출하고, SAM 2 알고리즘으로 픽셀 단위의 정밀한 마스크를 생성합니다. 이후 OpenCV의 침식(Erosion) 및 블러(Blur) 연산을 통해 테두리 노이즈를 완벽하게 제거하여 고품질 RGBA 이미지를 추출합니다.

### 🔹 1-2. 고품질 3D 메쉬 복원 엔진 (Generative & Reconstructive 3D)

- **Mode A (단일 이미지):** `TRELLIS` 엔진을 사용하여 단 한 장의 사진만으로 보이지 않는 뒷면까지 다각도로 상상(Hallucination)하여 3D 메쉬를 생성합니다.
- **Mode B (비디오/GIF/다중 이미지):** `InstantMesh LRM` 기반으로 동작합니다. 업로드된 미디어에서 Laplacian Variance(선명도) 기반으로 최적의 프레임을 자동 추출하고, 다각도 데이터를 분석해 왜곡 없는 정교한 3D 구조를 복원합니다.
- **Surface Optimization:** Open3D의 `Taubin Smoothing` 알고리즘을 적용하여 디테일(재봉선, 질감 등)은 보존하면서 3D 스캔 특유의 노이즈만 효과적으로 평탄화합니다.

### 🔹 1-3. 인터랙티브 가상 쇼룸 (Three.js WebGL Rendering)

- **Tech:** `Three.js`, `OrbitControls`, `TransformControls`, `HDRI (RoomEnvironment)`
- **Feature:** 생성된 3D 객체를 `Port 8502` 기반의 가상 공간에 즉시 렌더링합니다. 사용자는 W/E/R 키를 통해 객체의 위치, 크기, 회전을 직관적으로 제어할 수 있으며, 가죽/금속성(Roughness, Metalness) 및 실시간 조명(Exposure) 조절과 시네마틱 턴테이블 기능을 제공합니다.

---

## 🔄 2. 시스템 아키텍처

본 시스템은 두 개의 독립된 포트(8501, 8502)에서 비동기적으로 동작하는 아키텍처를 가집니다.

```
[ User Input (app.py :8501) ]
      ├─ Mode A: Single Image
      └─ Mode B: Video/GIF/Images ──> Smart Keyframe Extraction
                │
[ Segmentation Pipeline ]
      └─ Rembg (Box) ──> SAM 2 (Masking) ──> OpenCV (Denoising)
                │
[ 3D Generation Engine (GPU Heavy) ]
      ├─ TRELLIS Pipeline (Mode A)
      └─ InstantMesh LRM (Mode B) ──> Adaptive Scale Padding
                │
[ Post-Processing & Archiving ]
      └─ Open3D Taubin Smoothing ──> Local Storage (/outputs/meshes/)
                │
[ Visualization (showroom.py :8502) ]
      └─ Three.js HDRI Environment ──> Transform Controls / Material Editor
```

---

## 📁 3. 디렉토리 구조

```
📦 image-to-3d-viewer
 ┣ 📂 checkpoints/           # SAM 2 등 AI 모델 가중치(.pt) 저장소
 ┣ 📂 outputs/
 ┃ ┗ 📂 meshes/              # 생성된 최종 3D 모델(.obj) 저장소
 ┣ 📂 pipeline/
 ┃ ┣ 📂 instantmesh_core/    # InstantMesh 외부 C++ 코어 모듈 (git clone 필요)
 ┃ ┣ 📜 instantmesh.py       # Mode B 3D 생성 파이프라인 래퍼
 ┃ ┗ 📜 trellis.py           # Mode A 3D 생성 파이프라인 래퍼
 ┣ 📜 app.py                 # 메인 AI 추론 UI (Streamlit Port: 8501)
 ┣ 📜 showroom.py            # Three.js 3D 렌더링 서버 (Streamlit Port: 8502)
 ┣ 📜 requirements.txt       # 의존성 패키지 명세서
 ┗ 📜 README.md              # 프로젝트 명세서
```

---

## 🛠️ 4. 설치 및 실행 가이드

본 프로젝트는 고성능 3D 렌더링 연산을 위해 NVIDIA GPU (CUDA) 및 Python 3.10 환경이 필수적으로 요구됩니다. 컴파일러 충돌 방지를 위해 아래 가이드를 엄격히 따라주세요.

### 📋 4-1. 사전 준비 (Prerequisites)

- Python 3.10.x (3.11 이상은 3D 컴파일러 연동 불가)
- NVIDIA CUDA Toolkit 12.1 (Visual Studio Integration 체크 해제)
- Visual Studio 2022 C++ Build Tools (설치 후 `x64 Native Tools Command Prompt for VS 2022` 터미널 사용 권장)

### ⚙️ 4-2. 환경 세팅 및 패키지 설치

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate

# 2. InstantMesh 코어 엔진 다운로드 (필수)
git clone https://github.com/TencentARC/InstantMesh.git pipeline/instantmesh_core

# 3. Pip 업그레이드 및 필수 코어 설치
python -m pip install --upgrade pip
pip install torch torchvision numpy wheel setuptools

# 4. 전체 의존성 설치
pip install -r requirements.txt --no-build-isolation
```

---

## 🚀 5. 사용 방법

터미널을 2개 열고 각각의 서버를 실행하여 마이크로서비스 형태로 사용합니다.

**Terminal 1 — AI Generation Server**

```bash
venv\Scripts\activate
streamlit run app.py --server.port 8501
```

**Terminal 2 — 3D Showroom Server**

```bash
venv\Scripts\activate
streamlit run showroom.py --server.port 8502
```

웹 브라우저에서 `http://localhost:8501`에 접속하여 이미지/영상을 업로드하고 3D 객체를 생성합니다.

생성이 완료되면 `http://localhost:8502`에서 3D 렌더링 쇼룸을 확인하고 마우스/키보드(W, E, R)로 조작할 수 있습니다.

---

## 🚨 6. 트러블슈팅

**Q: `ModuleNotFoundError: No module named 'triton'` 에러가 발생합니다.**

A: Windows 환경에서 xformers 사용 시 발생하는 정상적인 로그입니다. 시스템이 자동으로 다른 어텐션(Attention) 가속기를 대체하여 사용하므로 프로세스에 지장이 없습니다.

**Q: 메쉬 생성 속도가 너무 느립니다.**

A: GPU VRAM이 부족하여 RAM 스왑이 일어날 가능성이 있습니다. app.py 구동 전 백그라운드의 불필요한 프로그램을 종료하십시오. 시스템은 psutil을 통해 가용 램을 파악하여 해상도(Resolution)를 동적으로 낮추는 방어 로직이 적용되어 있습니다.

**Q: InstantMesh 실행 시 Subprocess Error가 발생합니다.**

A: `pipeline/instantmesh_core` 폴더가 정상적으로 Clone 되었는지 확인하고, 터미널 로그의 상세 CUDA 컴파일 에러를 확인하세요. (VS 2022 C++ Build Tools 환경 변수 누락이 주원인입니다.)

---

## ⚠️ 7. 프로젝트 한계점 및 향후 개선 과제

현재 파이프라인이 가진 구조적 한계와 이를 극복하기 위한 향후 연구 방향입니다.

### 🛑 7-1. 단일 이미지 기반 복원(Mode A)의 기하학적 유추 한계

현재 Mode A에서 사용되는 생성형 3D AI(TRELLIS)는 '현실 세계의 물리적 객체(가구, 신발, 식기 등)'에 대한 학습 데이터를 기반으로 깊이(Depth)와 후면을 유추합니다.

- **문제점 (The Cartoon/2D Illustration Issue):** 피카츄와 같은 2D 애니메이션 캐릭터나 일러스트, 비현실적이고 극단적인 비대칭 구조를 가진 이미지를 입력할 경우, AI가 입체적인 부피감을 올바르게 상상하지 못합니다. 이로 인해 메쉬(Mesh)가 납작하게 눌리거나(Flat-distortion), 텍스처가 기괴하게 붕괴하는 현상이 발생합니다.
- **해결 방향:** 비현실적인 캐릭터나 2D 일러스트 전용으로 파인튜닝(Fine-tuning)된 LRM(Large Reconstruction Model) 모델을 파이프라인에 추가 연동하거나, 다중 프레임(Mode B)을 사용하여 입체적 단서를 강제로 제공하는 방식으로 개선할 예정입니다.

### 🛑 7-2. 측면(Side-view) 단일 프레임 의존성에 의한 할루시네이션(Hallucination)

- **문제점:** Mode B에서 GIF/비디오를 분석할 때, 추출된 메인 프레임(Frame 1)이 '완벽한 정면'이 아닌 '측면'일 경우, AI가 측면의 형태를 객체의 정면으로 오인하여 양쪽으로 대칭 복사(데칼코마니 현상, ex. 신발 코가 하트 모양으로 갈라짐)하는 할루시네이션이 발생합니다.
- **해결 방향:** 현재는 UI 상에서 사용자가 직접 최적의 프레임을 스왑(Swap)하도록 유도하여 방어하고 있으나, 향후 CLIP 또는 DINOv2와 같은 비전 모델을 도입하여 가장 정면에 가까운 프레임을 AI가 스스로 판단하여 1번 프레임으로 배치하는 'Auto-Pose Detection' 알고리즘을 개발할 계획입니다.
