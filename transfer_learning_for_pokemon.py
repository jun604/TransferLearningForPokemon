import cv2 as cv
import torch
import torchvision.models as models
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
top_k = 5

def GetModel(imgfile):
    # Load the pre-trained ResNet model and its weights
    weights = models.ResNet34_Weights.DEFAULT
    model = models.resnet34(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 150) 
    model = model.to(device)
    # Read the given image
    img_bgr = cv.imread(imgfile)
    assert img_bgr is not None, 'Cannot read the given image'

    # Preprocess the image and prepare the input tensor
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)      # Convert (H, W, C) to (C, H, W)
    preprocess = weights.transforms()
    input_batch = preprocess(img_tensor).unsqueeze(0).to(device) # Convert (C, H, W) to (1, C, H, W)

    return SimpleNamespace(
        model=model, 
        weights=weights, 
        input_batch=input_batch, 
        preprocess=preprocess)

def WhatIsThisPokemon(model, input_batch, top_k=5):
    # Perform inference and get the top-k predictions
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
    return zip(top_probs[0].cpu().tolist(), top_indices[0].cpu().tolist())


img_file = 'sample.jpeg'  # 본인의 이미지 파일 이름으로 바꾸세요.
model1 = GetModel(img_file)
predictions = WhatIsThisPokemon(model1.model, model1.input_batch, top_k)


name_map = {}
# 'mapping.txt'는 본인의 파일 이름으로 바꾸세요.
with open('PokemonKorean.txt', 'r', encoding='utf-8') as f:
    for line in f:
        # 빈 줄은 건너뜁니다.
        if not line.strip():
            continue
        # 1. 쉼표로 분리
        parts = [x.strip() for x in line.split(',')]
        # 2. [한글이름, 영어이름] 구조가 맞는지 확인
        if len(parts) == 2:
            kor_name = parts[0]
            eng_name = parts[1]
            # 영어 이름을 Key로, 한글 이름을 Value로 저장
            # (모델의 dataset.classes가 영어이므로 영어를 Key로 잡아야 찾기 쉽습니다)
            name_map[eng_name] = kor_name
dataset = datasets.ImageFolder('dataset/PokemonData', transform=model1.preprocess)
# Print the top-k predicted categories and their probabilities
categories = dataset.classes
print(f'Image: {img_file}')
print(f'Top-{top_k} predictions:')
for rank, (prob, index) in enumerate(predictions):
    pokemon_name = name_map.get(categories[index], categories[index])
    print(f'{rank + 1}. {pokemon_name} ({prob * 100:.2f}%)')