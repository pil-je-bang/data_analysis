import torch
import torch.nn as nn  #뉴럴네트워크
import matplotlib.pyplot as plt

# 학습 데이터 생성
x = torch.linspace(0, 10, 100).unsqueeze(1)  # 입력 데이터
y = 2*x + 1 + torch.randn(100, 1)  # 정답 레이블 (기울기: 2, 절편: 1)

# 모델 정의
model = nn.Linear(1, 1)  # 선형 모델 (입력 차원: 1, 출력 차원: 1)

# 손실 함수 정의
criterion = nn.MSELoss()

# 옵티마이저 정의
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #러닝레이트

# 학습
num_epochs = 1000  # 에폭 수

for epoch in range(num_epochs):
    # Forward 계산
    outputs = model(x)
    
    # 손실 계산
    loss = criterion(outputs, y)
    
    # Backward 계산 및 경사 하강
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 로그 출력
    if (epoch+1) % 100 == 0:
        print(f'에폭 [{epoch+1}/{num_epochs}], 손실: {loss.item():.4f}')

# 학습된 모델의 예측 결과 확인
predicted = model(x)

# 그래프 그리기
plt.scatter(x, y, label='real_data')
plt.plot(x, predicted.detach().numpy(), color='red', label='predicted_result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()