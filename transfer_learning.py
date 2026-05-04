import cv2 as cv
import torch
import torchvision.models as models
from torchvision import datasets, transforms

# Initialize control parameters
img_file = 'sample.jpeg'
top_k = 5

# Read the given image
img_bgr = cv.imread(img_file)
assert img_bgr is not None, 'Cannot read the given image'

# Load the pre-trained ResNet model and its weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = models.ResNet34_Weights.DEFAULT
# 1. 사전 학습된 모델 로드
model = models.resnet34(weights=weights)
# 2. 마지막 층의 입력 피처 수 확인 (512)
num_ftrs = model.fc.in_features
# 3. 출력 클래스 수를 150(포켓몬 수)으로 변경
model.fc = torch.nn.Linear(num_ftrs, 150) 
# 4. 장치로 전송
model = model.to(device)

# Preprocess the image and prepare the input tensor
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)      # Convert (H, W, C) to (C, H, W)
preprocess = weights.transforms()
input_batch = preprocess(img_tensor).unsqueeze(0).to(device) # Convert (C, H, W) to (1, C, H, W)
dataset = datasets.ImageFolder('dataset/PokemonData', transform=preprocess)

# Perform inference and get the top-k predictions
model.eval()
with torch.no_grad():
    output = model(input_batch)
    probabilities = torch.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

# Print the top-k predicted categories and their probabilities
categories = dataset.classes
print(f'Image: {img_file}')
print(f'Top-{top_k} predictions:')
for rank, (prob, index) in enumerate(zip(top_probs[0].cpu().tolist(), top_indices[0].cpu().tolist())):
    print(f'{rank + 1}. {categories[index]} ({prob * 100:.2f}%)')