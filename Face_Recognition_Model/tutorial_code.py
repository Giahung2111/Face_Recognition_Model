# 1. Setup

# 1.1 Install Dependencies
# Cài đặt PyTorch qua pip (thay vì TensorFlow)
# Chạy trên terminal: pip install torch torchvision torchaudio opencv-python matplotlib

# 1.2 Import Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# 1.3 Set GPU Growth
# PyTorch tự động quản lý bộ nhớ GPU, không cần thiết lập như TensorFlow

# 1.4 Create Folder Structures
# Thiết lập các đường dẫn thư mục
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# 2. Collect Positives and Anchors

# 2.1 Untar Labelled Faces in the Wild Dataset
# Giải nén dataset LFW và di chuyển ảnh vào thư mục negative
# Chạy lệnh hệ thống (giữ nguyên từ TensorFlow)
# !tar -xf lfw.tgz
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

# 2.2 Collect Positive and Anchor Classes
import uuid

# Kết nối với webcam để thu thập ảnh
cap = cv2.VideoCapture(0)  # Có thể cần điều chỉnh index camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Cắt khung ảnh thành 250x250px
    frame = frame[120:120+250, 200:200+250, :]

    # Thu thập ảnh anchor khi nhấn phím 'a'
    if cv2.waitKey(1) & 0xFF == ord('a'):
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    # Thu thập ảnh positive khi nhấn phím 'p'
    if cv2.waitKey(1) & 0xFF == ord('p'):
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    # Hiển thị khung ảnh
    cv2.imshow('Image Collection', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 2.x NEW - Data Augmentation
# Hàm tăng cường dữ liệu sử dụng torchvision.transforms
def data_aug(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.02, contrast=0.4, saturation=0.1),
        transforms.ToTensor()
    ])
    data = []
    for _ in range(9):  # Tạo 9 phiên bản tăng cường
        aug_img = transform(img)
        data.append(aug_img)
    return data

# Áp dụng tăng cường cho một ảnh anchor
img_path = os.path.join(ANC_PATH, 'some_image.jpg')  # Thay bằng tên ảnh thực tế
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
augmented_images = data_aug(img)

# Lưu các ảnh tăng cường
for aug_img in augmented_images:
    aug_img = aug_img.permute(1, 2, 0).numpy() * 255  # Chuyển về định dạng numpy
    aug_img = aug_img.astype(np.uint8)
    cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), aug_img)

# 3. Load and Preprocess Images

# 3.1 Get Image Directories
from glob import glob

# Lấy danh sách file ảnh (giới hạn 3000 ảnh mỗi loại)
anchor_files = glob(os.path.join(ANC_PATH, '*.jpg'))[:3000]
positive_files = glob(os.path.join(POS_PATH, '*.jpg'))[:3000]
negative_files = glob(os.path.join(NEG_PATH, '*.jpg'))[:3000]

# 3.2 Preprocessing - Scale and Resize
# Hàm tiền xử lý ảnh
def preprocess(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))  # Resize về 100x100
    img = img / 255.0  # Chuẩn hóa về [0, 1]
    return img

# 3.3 Create Labelled Dataset
# Tạo lớp Dataset tùy chỉnh cho Siamese Network
class SiameseDataset(Dataset):
    def __init__(self, anchor_files, positive_files, negative_files):
        self.anchor_files = anchor_files
        self.positive_files = positive_files
        self.negative_files = negative_files

    def __len__(self):
        return len(self.anchor_files) * 2  # Tổng số cặp positive và negative

    def __getitem__(self, idx):
        if idx < len(self.anchor_files):  # Cặp anchor-positive
            anchor = preprocess(self.anchor_files[idx])
            positive = preprocess(self.positive_files[idx])
            label = 1.0
        else:  # Cặp anchor-negative
            anchor = preprocess(self.anchor_files[idx - len(self.anchor_files)])
            negative = preprocess(self.negative_files[idx - len(self.anchor_files)])
            label = 0.0
        return (torch.tensor(anchor, dtype=torch.float32).permute(2, 0, 1),  # Chuyển sang định dạng PyTorch
                torch.tensor(positive if label == 1.0 else negative, dtype=torch.float32).permute(2, 0, 1),
                torch.tensor(label, dtype=torch.float32))

dataset = SiameseDataset(anchor_files, positive_files, negative_files)

# 3.4 Build Train and Test Partition
# Chia tập dữ liệu thành train và test
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 4. Model Engineering

# 4.1 Build Embedding Layer
# Định nghĩa mạng nhúng (Embedding Network)
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x

embedding_net = EmbeddingNet()

# 4.2 Build Distance Layer
# Lớp tính khoảng cách L1
class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, input_embedding, validation_embedding):
        return torch.abs(input_embedding - validation_embedding)

# 4.3 Make Siamese Model
# Định nghĩa mô hình Siamese hoàn chỉnh
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = embedding_net
        self.l1_dist = L1Dist()
        self.fc = nn.Linear(4096, 1)  # Đầu ra nhị phân
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_img, validation_img):
        input_embedding = self.embedding_net(input_img)
        validation_embedding = self.embedding_net(validation_img)
        distances = self.l1_dist(input_embedding, validation_embedding)
        output = self.sigmoid(self.fc(distances))
        return output

siamese_model = SiameseNetwork(embedding_net)

# 5. Training

# 5.1 Setup Loss and Optimizer
# Sử dụng Binary Cross Entropy và Adam optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(siamese_model.parameters(), lr=1e-4)

# 5.2 Establish Checkpoints
# Thiết lập thư mục lưu checkpoint
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 5.3 Build Train Step Function
# Hàm thực hiện một bước huấn luyện
def train_step(batch):
    input_img, validation_img, label = batch
    label = label.unsqueeze(1)  # Thêm chiều để khớp với output

    optimizer.zero_grad()
    output = siamese_model(input_img, validation_img)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    return loss.item()

# 5.4 Build Training Loop
# Vòng lặp huấn luyện với tính toán precision và recall
from sklearn.metrics import precision_score, recall_score

def train(train_loader, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print(f'\n Epoch {epoch}/{EPOCHS}')
        total_loss = 0
        y_true = []
        y_pred = []

        for batch in train_loader:
            loss = train_step(batch)
            total_loss += loss

            # Dự đoán để tính precision và recall
            input_img, validation_img, label = batch
            with torch.no_grad():
                output = siamese_model(input_img, validation_img)
            pred = (output > 0.5).float().squeeze().numpy()
            y_pred.extend(pred)
            y_true.extend(label.numpy())

        avg_loss = total_loss / len(train_loader)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        print(f'Loss: {avg_loss}, Precision: {precision}, Recall: {recall}')

        # Lưu checkpoint mỗi 10 epoch
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': siamese_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(checkpoint_dir, f'ckpt_epoch_{epoch}.pt'))

# 5.5 Train the model
EPOCHS = 50
train(train_loader, EPOCHS)

# 6. Evaluate Model

# 6.1 Import Metrics
# Đã import precision_score và recall_score ở trên

# 6.2 Make Predictions
# Lấy một batch từ test_loader và dự đoán
test_batch = next(iter(test_loader))
input_img, validation_img, y_true = test_batch
with torch.no_grad():
    y_hat = siamese_model(input_img, validation_img)
y_hat = (y_hat > 0.5).float().squeeze().numpy()

# 6.3 Calculate Metrics
# Tính precision và recall cho batch
precision = precision_score(y_true, y_hat)
recall = recall_score(y_true, y_hat)
print(f'Precision: {precision}, Recall: {recall}')

# 6.4 Viz Results
# Hiển thị ảnh đầu vào và xác thực
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(input_img[0].permute(1, 2, 0).numpy())  # Chuyển về định dạng HWC
plt.subplot(1, 2, 2)
plt.imshow(validation_img[0].permute(1, 2, 0).numpy())
plt.savefig('result.png')  # Lưu ảnh thay vì hiển thị trực tiếp

# 7. Save Model
# Lưu mô hình sau khi huấn luyện
torch.save(siamese_model.state_dict(), 'siamesemodelv2.pth')

# 8. Real Time Test

# 8.1 Verification Function
# Hàm xác thực trong thời gian thực
def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        input_img = torch.tensor(input_img).permute(2, 0, 1).unsqueeze(0).float()
        validation_img = torch.tensor(validation_img).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            result = model(input_img, validation_img).item()
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold
    return results, verified

# 8.2 OpenCV Real Time Verification
# Xác thực thời gian thực với webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[120:120+250, 200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Kích hoạt xác thực khi nhấn 'v'
    if cv2.waitKey(10) & 0xFF == ord('v'):
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        results, verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()