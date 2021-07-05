#### 📚 Stanford cs231n  
#### 🚩 Euron AI team 황시은 9주차 예습  
### Lecture 10 | Recurrent Neural Networks

Recurrent Neural Networks(RNN)
: 기본적인 one-to-one valnilla NN이 연속적으로 연결된 구조를 바탕으로 한다.  

![](https://media.vlpt.us/images/ryuni/post/0a335a93-f372-4323-89fa-77194d9e0d27/86EFA26A-6AD1-4066-96DA-6ADCEF8B29EA.jpeg)

1. one-to-many: 이미지를 입력했을 때 그를 설명하는 단어들을 만들어내는 image captioning에서 많이 사용함  
2. many-to-one: 연속적인 단어 입력시 감정을 추출하는 sentiment classification에서 많이 사용  
3. many-to-many: 번역 모델  
4. many-to-many: 가장 오른쪽 모델, 실시간 영상 분류  

RNN의 기본 구조  
: hidden state에 저장된 값을 다음 단계에서 사용하며, 이 때 가중치는 언제나 동일하다.  
![](https://media.vlpt.us/images/ryuni/post/d8a1312e-3561-4d43-8120-25f6dcb1bafe/2D0631E2-DB49-4833-81BE-EE62187A3CE5.jpeg)


Truncated Backpropagation through time  
: 전체 문장을 잘게 쪼개서 단위 별로 loss를 계산하는 방식.

LSTM(Long Short Term Memory)
: 보통 성능을 높이기 위해 RNN을 여러층 쌓는데 이때 모델이 커지면 거리가 떨어진 정보끼리 gradient가 전달 잘 안될 수 있음. 이 문제를 해결하는 것이 LSTM  

Image Captioning
: CNN의 출력 값을 RNN의 입력으로 사용하고, 이 정보로부터 문장을 만들어 내는 방법이다.  

Image Captioning with Attention
: 특정 부분에 집중하여 볼 수 있음. 각 벡터가 공간 정보를 가지고 있는 grid of vector를 만들어 낸다. 
![](https://fbdp1202.github.io/assets/img/dev/mldl/cs231n/lecture10/cs231n-10-026-RNN_Example_Image_Captioning_Attention_04.png)
