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