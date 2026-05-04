import cv2 as cv
import torch
import torchvision.models as models
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.utils.data import random_split
import torch_directml
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if torch_directml.is_available():
    device = torch_directml.device()
    print("AMD GPU (DirectML)를 사용합니다.")
else:
    device = torch.device('cpu')
    print("GPU를 찾을 수 없어 CPU를 사용합니다.")
top_k = 5

def GetModel(ResNetModel=models.resnet34, ResNetWeights=models.ResNet34_Weights.DEFAULT):
    # Load the pre-trained ResNet model and its weights
    weights = ResNetWeights
    model = ResNetModel(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 150) 
    model = model.to(device)

    # Preprocess the image and prepare the input tensor
    preprocess = weights.transforms()

    return SimpleNamespace(
        model=model,
        weights=weights,
        preprocess=preprocess)

def WhatIsThisPokemon(Model, imgfile , top_k=5):
    # Read the given image
    img_bgr = cv.imread(imgfile)
    assert img_bgr is not None, 'Cannot read the given image'
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)      # Convert (H, W, C) to (C, H, W)
    input_batch = Model.preprocess(img_tensor).unsqueeze(0).to(device) # Convert (C, H, W) to (1, C, H, W)
    # Perform inference and get the top-k predictions
    Model.model.eval()
    with torch.no_grad():
        output = Model.model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
    return zip(top_probs[0].cpu().tolist(), top_indices[0].cpu().tolist())

def TrainModel(Model, train_loader, epochs=5):
    # 1. 학습에 필요한 도구 세팅
    criterion = nn.CrossEntropyLoss()
    # Model1.model에 접근하여 파라미터를 최적화기에 전달
    optimizer = optim.Adam(Model.model.parameters(), lr=0.001)
    
    history = []

    Model.model.train() # 학습 모드 전환
    print("학습 시작...")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()   # 기울기 초기화
            outputs = Model.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()         # 역전파
            optimizer.step()        # 가중치 업데이트
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    print("학습 완료!")
    return history

def RandomSplitDataset(dataset, train_ratio=0.8):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    return SimpleNamespace(
        train_data=train_data, test_data=test_data)

# 로더 생성
def CreateDataLoader(Dataset, batch_size=32):
    train_loader = DataLoader(Dataset.train_data, batch_size, shuffle=True)
    test_loader = DataLoader(Dataset.test_data, batch_size, shuffle=False)
    return SimpleNamespace(
        train_loader=train_loader, test_loader=test_loader)

def Performance(Model, test_loader):
    Model.model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n--- 성능 평가 시작 ---")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = Model.model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Precision, Recall, F1-score 계산 (150개 클래스이므로 'macro' 또는 'weighted' 권장)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"Test Precision (Weighted): {precision:.4f}")
    print(f"Test Recall (Weighted): {recall:.4f}")
    print(f"Test F1-Score (Weighted): {f1:.4f}")
    
    # 상세 보고서 (클래스별 지표가 궁금할 때 사용)
    # print(classification_report(all_labels, all_preds, target_names=categories))
    
    return SimpleNamespace(
        precision=precision, recall=recall, f1=f1)

def SaveLearningCurve(History):
    # 각 모델의 학습 기록을 그래프에 추가
    for i, history in enumerate(History):
        plt.plot(range(1, len(history) + 1), history, label=f'Model {i+1}')
    filename = f'learning_curve.png'
    plt.title(f"Training Loss Comparison") # 제목
    plt.xlabel('Epochs')                 # x축 이름
    plt.ylabel('Loss')                   # y축 이름
    plt.legend()                         # 범례(Model 1, 2...) 표시
    plt.grid(True)                       # 격자 표시
    
    # 이미지 파일로 저장
    plt.savefig(filename)
    plt.close()                          # 메모리 확보를 위해 창 닫기
    print(f"그래프가 {filename}으로 저장되었습니다.")

def SavePerformance(Result):
    # 데이터 정리
    df = pd.DataFrame(Result) # [{'model': 'M1', 'precision': 0.8...}, ...]
    metrics = ['precision', 'recall', 'f1']
    
    # 각 모델의 학습 기록을 그래프에 추가
    for i, result in enumerate(Result):
        plt.plot(range(1, len(result) + 1), result, label=f'Model {i+1}')
    filename = f'performance_comparison.png'
    plt.title(f"Performance Comparison") # 제목
    plt.xlabel('Performance')                 # x축 이름
    plt.ylabel('Score')                   # y축 이름
    plt.ylim(0, 1)                        # y축 범위 0~1로 고정
    plt.legend()                         # 범례(Model 1, 2...) 표시
    #plt.grid(True)                       # 격자 표시
    # 점수 텍스트 표시 (각 점 위에 수치 기입)
    for i in range(len(df)):
        plt.text(x=df.Metric.iloc[i], y=df.Score.iloc[i] + 0.02, 
                 s=f'{df.Score.iloc[i]:.4f}', 
                 ha='center', fontsize=10, fontweight='semibold')
    
    # 이미지 파일로 저장
    plt.savefig(filename)
    plt.close()                          # 메모리 확보를 위해 창 닫기
    print(f"그래프가 {filename}으로 저장되었습니다.")



img_file = 'Sample.jpg'
dataset = datasets.ImageFolder('dataset/PokemonData')
categories = dataset.classes
History = []
Result = []
Predictions = []

# ResNet34 모델로 실험
Model1 = GetModel()
Dataset1 = RandomSplitDataset(datasets.ImageFolder('dataset/PokemonData', transform=Model1.preprocess))
DataLoader1 = CreateDataLoader(Dataset1)
history1 = TrainModel(Model1, DataLoader1.train_loader, epochs=5)
History.append(history1)
result1 = Performance(Model1, DataLoader1.test_loader)
Result.append(result1)
predictions1 = WhatIsThisPokemon(Model1, img_file, top_k)
Predictions.append(predictions1)

Model2 = GetModel()
Dataset2 = RandomSplitDataset(datasets.ImageFolder('dataset/PokemonData', transform=Model2.preprocess))
DataLoader2 = CreateDataLoader(Dataset2)
for param in Model2.model.parameters():
    param.requires_grad = False  # 모든 층의 공부를 중단시킴
# 마지막 층(우리가 바꾼 150개 층)만 다시 공부하도록 설정
for param in Model2.model.fc.parameters():
    param.requires_grad = True
history2 = TrainModel(Model2, DataLoader2.train_loader, epochs=5)
History.append(history2)
result2 = Performance(Model2, DataLoader2.test_loader)
Result.append(result2)
predictions2 = WhatIsThisPokemon(Model2, img_file, top_k)
Predictions.append(predictions2)


# ResNet18 모델로 실험
Model3 = GetModel(models.resnet18, models.ResNet18_Weights.DEFAULT)
Dataset3 = RandomSplitDataset(datasets.ImageFolder('dataset/PokemonData', transform=Model3.preprocess))
DataLoader3 = CreateDataLoader(Dataset3)
history3 = TrainModel(Model3, DataLoader3.train_loader, epochs=5)
History.append(history3)
result3 = Performance(Model3, DataLoader3.test_loader)
Result.append(result3)
predictions3 = WhatIsThisPokemon(Model3, img_file, top_k)
Predictions.append(predictions3)

Model4 = GetModel(models.resnet18, models.ResNet18_Weights.DEFAULT)
Dataset4 = RandomSplitDataset(datasets.ImageFolder('dataset/PokemonData', transform=Model4.preprocess))
DataLoader4 = CreateDataLoader(Dataset4)
for param in Model4.model.parameters():
    param.requires_grad = False  # 모든 층의 공부를 중단시킴
# 마지막 층(우리가 바꾼 150개 층)만 다시 공부하도록 설정
for param in Model4.model.fc.parameters():
    param.requires_grad = True
history4 = TrainModel(Model4, DataLoader4.train_loader, epochs=5)
History.append(history4)
result4 = Performance(Model4, DataLoader4.test_loader)
Result.append(result4)
predictions4 = WhatIsThisPokemon(Model4, img_file, top_k)
Predictions.append(predictions4)


SaveLearningCurve(History)
SavePerformance(Result)


name_map = {}
# 'mapping.txt'는 본인의 파일 이름으로 바꾸세요.
with open('PokemonKorean.txt', 'r', encoding='utf-8') as f:
    for line in f:
        # 빈 줄은 건너뜀
        if not line.strip():
            continue
        # 쉼표로 분리
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 2:
            kor_name = parts[0]
            eng_name = parts[1]
            name_map[eng_name] = kor_name

# Print the top-k predicted categories and their probabilities
print(f'Image: {img_file}')
print(f'Model1 - Top-{top_k} predictions:')
for rank, (prob, index) in enumerate(predictions1):
    pokemon_name = name_map.get(categories[index], categories[index])
    print(f'{rank + 1}. {pokemon_name} ({prob * 100:.2f}%)')
    
print(f'Model2 - Top-{top_k} predictions:')
for rank, (prob, index) in enumerate(predictions2):
    pokemon_name = name_map.get(categories[index], categories[index])
    print(f'{rank + 1}. {pokemon_name} ({prob * 100:.2f}%)')

print(f'Model3 - Top-{top_k} predictions:')
for rank, (prob, index) in enumerate(predictions3):
    pokemon_name = name_map.get(categories[index], categories[index])
    print(f'{rank + 1}. {pokemon_name} ({prob * 100:.2f}%)')

print(f'Model4 - Top-{top_k} predictions:')
for rank, (prob, index) in enumerate(predictions4):
    pokemon_name = name_map.get(categories[index], categories[index])
    print(f'{rank + 1}. {pokemon_name} ({prob * 100:.2f}%)')