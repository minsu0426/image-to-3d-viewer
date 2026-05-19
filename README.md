🚀 image-to-3d-viewer
SAM 2(Segment Anything 2)와 TripoSR을 결합한 지능형 2D-to-3D 변환 및 가상 쇼룸 시스템

1. 프로젝트 개요 (Project Overview)
본 프로젝트는 컴퓨터 비전의 최신 기술을 활용하여 단일 2D 이미지로부터 고품질의 3D 에셋을 생성하고 이를 가상 공간에서 관리하는 인터랙티브 웹 애플리케이션입니다.
복잡한 3D 모델링 지식 없이도 사진 한 장으로 정교한 3D 메쉬를 생성할 수 있도록 하는 것이 목표입니다.

2. 핵심 기능 (Core Features)
정밀 객체 세그멘테이션 (SAM 2 Integration): 사용자의 클릭 프롬프트를 바탕으로 이미지 내 특정 객체를 픽셀 단위로 정교하게 분리 및 배경 제거(RGBA 추출).

인스턴트 3D 메쉬 생성 (TripoSR Engine): 분리된 2D 객체 이미지를 입력받아 수 초 내에 텍스처가 포함된 3D 메쉬(.obj, .glb) 생성.

인터랙티브 3D 가상 쇼룸 (3D Visualization): 생성된 3D 객체들을 브라우저 내 3D 공간에 배치하고, 마우스 인터랙션을 통한 360도 회전, 확대 및 축소 기능 제공.

로컬 아카이빙 시스템: 생성된 3D 모델 파일 및 메타데이터를 로컬 세션에 저장하여 목록화하는 관리 기능.

3. 시스템 파이프라인 (System Pipeline)
시스템은 다음과 같은 단계별 데이터 흐름을 통해 동작합니다.

Input Phase: 사용자가 웹 UI를 통해 2D 이미지를 업로드하고 관심 객체를 지정(Click/Box).

Segmentation Phase (SAM 2): 지정된 좌표를 기반으로 객체의 마스크를 생성하고, 원본 이미지에서 해당 객체만 추출하여 투명 배경 이미지로 변환.

Reconstruction Phase (TripoSR): 추출된 이미지를 3D 생성 모델에 전달하여 깊이(Depth) 및 기하학적 구조를 추정한 후 3D 메쉬 데이터를 생성.

Rendering Phase: 생성된 메쉬 파일을 Three.js 또는 PyVista 엔진을 통해 웹 화면에 실시간으로 시각화 및 배치.

## 📜 출처 및 오픈소스 크레딧 (Acknowledgments & References)

본 텀프로젝트는 컴퓨터 비전 분야의 최신 오픈소스 파운데이션 모델들을 기반으로 시스템 통합을 수행한 연구 및 개발 결과물입니다. 핵심 파이프라인 구현을 위해 아래의 훌륭한 오픈소스 프로젝트와 연구 자산을 활용 및 참조하였습니다.

### 1. 사용 오픈소스 라이브러리 및 모델
- **SAM 2 (Segment Anything Model 2)**: Meta AI에서 개발한 Zero-shot 인스턴스 세그멘테이션 모델을 활용하여 이미지 내 2D 객체 추출 및 배경 제거(RGBA 변환) 파이프라인을 구축하였습니다.
  - Repository: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **TripoSR**: VAST-AI Research와 Stability AI에서 공동 개발한 단일 이미지 기반 초고속 3D Reconstruction 모델을 활용하여 2D 객체의 3D 메쉬(.obj) 복원을 수행하였습니다.
  - Repository: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- **Three.js**: 웹 브라우저 환경에서 별도의 플러그인 없이 고성능 3D 그래픽을 렌더링하기 위해 Three.js(r128) 라이브러리를 임베딩하여 인터랙티브 가상 쇼룸을 구현하였습니다.

### 2. 연구 인용 (BibTeX)

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Grigorev, Valentin and Karyakin, Karttikeya and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}

@article{tochilkin2024triposr,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Panchenko, Maxim and Sanyal, Soubhik and others},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}

#$💡 실행 안내
AI 모델 가중치 파일은 용량 문제로 GitHub에 포함되어 있지 않습니다. 
하지만 코드를 최초 실행할 때 `app.py`가 알아서 모델 파일을 다운로드하여 `checkpoints/` 폴더에 저장하므로, 별도의 설정 없이 바로 실행하시면 됩니다!