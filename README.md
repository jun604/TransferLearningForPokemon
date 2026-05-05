# Transfer Learning for Pokemon
사전 학습된 ResNet 모델을 활용하여 150종의 포켓몬 이미지를 분류하는 전이 학습(Transfer Learning) 프로그램</br>
PyTorch 프레임워크 기반의 모델 설계와 DirectML을 이용한 AMD GPU 가속 학습 환경 구현</br>
Fine-tuning과 Full Training 방식의 성능 비교를 통해 최적의 포켓몬 분류 모델 산출

## 기능
1. 데이터셋 전처리 및 동적 로딩
   - Kaggle 포켓몬 데이터셋을 ImageFolder 구조로 로드하고 8:2 비율로 Train/Test 데이터 분할
   - ResNet 모델별 전용 transforms를 적용하여 입력 이미지의 스케일 및 정규화 최적화
2. 다중 모델 구조 설계 및 학습 전략
   - ResNet34 및 ResNet18 모델을 기반으로 150개의 클래스를 분류하도록 Fully Connected Layer 재설계
   - 가중치를 모두 업데이트하는 'Full Training'과 마지막 층만 학습시키는 'Fine-tuning(Frozen)' 전략 병행
3. DirectML 기반 하드웨어 가속
   - torch_directml 라이브러리를 활용하여 NVIDIA GPU 외에도 AMD Radeon GPU 등 다양한 하드웨어에서 가속 학습 지원
   - GPU 가용 상태를 자동 탐색하여 최적의 연산 장치(Device) 할당 로직 구현
4. 학습 로그 분석 및 시각화
   - 에폭(Epoch)별 Loss 변화를 기록하여 learning_curve.png로 저장함으로써 모델의 수렴 과정 모니터링
   - 모델별 Precision, Recall, F1-Score를 산출하고 performance_comparison.png를 통해 정량적 성능 지표 비교
5. 실시간 추론 및 한글 이름 매핑
   - 학습된 .pth 가중치를 로드하여 임의의 입력 이미지(Sample.jpg)에 대한 Top-5 예측 결과 산출
   - 영어 클래스 명칭을 PokemonKorean.txt와 매핑하여 사용자 친화적인 한 포켓몬 이름 출력 지원

### 실행 결과
- pokemon_resnet18_full.pth, pokemon_resnet18_finetuned.pth
  + resnet18로 학습 완료된 모델
  + resnet34의 경우 용량이 초과되어 올리지 못함
- learning_curve.png: 모델별 학습 손실(Loss) 비교 그래프
- performance_comparison.png: 주요 평가지표(Precision, Recall, F1) 성능 비교 차트
  + Model1(ResNet34 + Full Training)
    - Precision = 0.8422
    - Recall = 0.7933
    - F1-Score = 0.7894
  + Model2(ResNet34 + Fine-tuning(Frozen))
    - Precision = 0.8502
    - Recall = 0.8079
    - F1-Score = 0.8081
  + Model3(ResNet18 + Full Training)
    - Precision = 0.8947
    - Recall = 0.8688
    - F1-Score = 0.8664
  + Model4(ResNet18 + Fine-tuning(Frozen))
    - Precision = 0.8417
    - Recall = 0.7991
    - F1-Score = 0.7966

# Pokemon Classifier Demo With streamlit
Streamlit 프레임워크를 이용하여 학습된 포켓몬 분류 모델을 웹 브라우저에서 테스트할 수 있는 데모 애플리케이션</br>
캐싱(Caching) 최적화를 통해 대용량 딥러닝 모델의 로딩 속도를 개선하고 실시간 추론 환경 제공

## 기능
1. 인터랙티브 모델 선택 및 로드
   - 사이드바를 통해 ResNet18/34 기반의 4가지 가중치 파일 중 원하는 모델을 실시간으로 선택 및 교체
   - @st.cache_resource 데코레이터를 적용하여 반복적인 모델 로딩으로 인한 리소스 낭비 방지
2. 드래그 앤 드롭 방식의 이미지 업로드
   - file_uploader를 통해 JPG, PNG 형식의 이미지를 간편하게 입력받아 추론 엔진으로 전달
   - 업로드된 이미지를 OpenCV 좌표계로 변환하여 실시간 미리보기 화면 출력
3. 사용자 중심의 결과 시각화
   - st.columns 레이아웃 분할을 통해 입력 이미지와 Top-5 예측 결과를 한눈에 비교 가능하도록 구성
   - 모델이 예측한 확률값을 st.progress 게이지 바로 표현하여 결과의 신뢰도를 직관적으로 전달
4. 한글 레이블 지원 및 UI 고도화
   - 외부 텍스트 파일을 참조하여 영문 레이블을 실시간으로 한글 명칭으로 변환하여 표시
   - st.divider 및 마크다운 형식을 활용하여 모바일 및 웹 환경에 최적화된 깔끔한 UI 디자인 구현
5. 학습 지표 대시보드 통합
   - 사이드바 체크박스를 통해 학습 과정에서 생성된 Loss 곡선 및 성능 비교 그래프를 데모 화면 내에서 즉시 확인 지원

### 실행 방법
- 터미널에서 py -m streamlit run Streamlit_for_pokemon.py 실행
- run_demo.bat 실행
