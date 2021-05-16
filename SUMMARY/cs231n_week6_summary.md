#### 📚 Stanford cs231n  
#### 🚩 Euron AI team 황시은 6주차 예습  
### Training Neural Networks I  

Activation Functions  
Neural Network에서 뉴런으로 전달된 입력값이 연산 후 최종적으로 활성화 함수와 연산되고 다음 뉴런으로 전달된다.  
![image](https://user-images.githubusercontent.com/61612117/118396932-b8f9fe00-b68c-11eb-8b84-b6371c953a34.png)

다양한 Activation functions들의 예시
![image](https://user-images.githubusercontent.com/61612117/118396847-5bfe4800-b68c-11eb-8e1a-72b2571aac0c.png)

- Sigmoid Function  
![image](https://user-images.githubusercontent.com/61612117/118396868-76382600-b68c-11eb-9a73-fe24d14f8654.png)
입력을 받아서 0-1 사이의 값으로 만든다.  
입력이 크면 1에 가깝고, 작으면 0에 가깝다. 
Sigmoid 함수의 문제점  
	* 포화된 뉴런이 gradient를 죽인다(vanishing gradient)  
	* x가 아주 크거나 작을 때 gradient가 0이 된다.   
	* 함수의 출력값이 0을 중심으로 하지 않는다(not zero-centered)  
	* exp()함수가 계산이 오래 걸린다  
- tanh Function  
![image](https://user-images.githubusercontent.com/61612117/118397059-3a519080-b68d-11eb-913d-45f0c02c337b.png)
입력값을 -1-1사이로 만들고, 0을 중심으로 한다. 역시 뉴런이 포화되었을 때 gradient vanishing  
- ReLU
![image](https://user-images.githubusercontent.com/61612117/118397102-666d1180-b68d-11eb-8d30-4a883d466da1.png)
양수일 때 뉴런이 포화되지 않아 그래디언트가 죽지 않고, 계산이 효율적이고 빠르다. 수렴 속도 또한 약 6배 빠르다.  
ReLU Function의 문제점
	* not zero-centered
	* 음수일 때 포화되기 때문에 절반의 gradient가 죽는다.  
	* 가중치의 초기화가 잘못되어 가중치 평면이 Data Cloud와 멀리 떨어져 있거나, learning rate가 너무 클 때 **dead RuLU** 발생한다. 
- Leaky ReLU, PReLU
![image](https://user-images.githubusercontent.com/61612117/118397194-d2e81080-b68d-11eb-8fc5-494e82384be8.png)
ReLU의 문제점을 해결한다. 뉴런이 포화되지 않으며 계산도 효율적이다. 수렴 속도도 빠르고 gradient 죽지 않는다!  
- EReLU
![image](https://user-images.githubusercontent.com/61612117/118397235-032faf00-b68e-11eb-9dbc-e32e2fc7d73a.png)
- Maxout Neuron  
![image](https://user-images.githubusercontent.com/61612117/118397263-1b9fc980-b68e-11eb-8f29-69508beb5d57.png)
ReLU와 Leaky ReLU를 일반화시킨 활성화 함수이다.  

결론  
![image](https://user-images.githubusercontent.com/61612117/118397285-383c0180-b68e-11eb-9a21-4c2ac4378e1f.png)


**Data Preprocessing**  
![](https://media.vlpt.us/images/guide333/post/0adc3118-8e27-41f5-be27-5897681b3003/Screenshot%20from%202021-02-05%2001-18-33.png)

1) zero-centered data: sigmoid 함수에서와 같은 이유. 입력이 모두 양수/음수이면 모든 뉴런이 양수/음수인 그래디언트를 얻는다.  
2) Normalized data: 모든 차원이 동일한 범위 안에 있게 하여 전부 동등한 기여를 하게 한다.  
- PCA, whitenening은 통계적 학습에 적합하므로 이미지 분석에서는 잘 사용하지 않는다.  
- 이미지 분석에서는 정규화가 필요 없고 zero-centered로만 만들면 된다. 이미 입력 이미지의 각 차원이 특정 범위 안에 들어있기 때문이다.  

**Weight Initialization**  
![](https://media.vlpt.us/images/guide333/post/0b0c9161-95ad-41f0-8e97-757f7ed03da3/Screenshot%20from%202021-02-05%2001-19-21.png)
![](https://media.vlpt.us/images/guide333/post/93867296-49f8-4dd5-94de-434ece24c31a/Screenshot%20from%202021-02-05%2001-19-31.png)  
초기화의 첫번째 방법: 임의의 작은 값으로 초기화 한다.  
		- 작은 네트워크에서는 작동하나 깊으면 문제 생긴다.  
		- 너무 작으면 사라지고, 너무 크면 포화되기 때문에 적절한 w를 구하는 것은 어렵다. 

Xavier initialization  
![](https://media.vlpt.us/images/guide333/post/554e5d1c-ede4-4206-a510-f130e31e0cf2/Screenshot%20from%202021-02-05%2001-20-13.png)

- 가우시안 표준 정규 분포에서 랜덤으로 뽑은 값을 '입력의 수'로 스케일링 한다.  
ReLU를 사용할 때 weight initialization  
![](https://media.vlpt.us/images/guide333/post/daea124a-9a26-40d0-858c-1dcde4c4bc02/Screenshot%20from%202021-02-05%2001-20-28.png)
dead ReLU 발생하여 잘 동작하지 않는다.  

![](https://media.vlpt.us/images/guide333/post/e7416b45-e138-47d0-94d1-adf23629d37d/Screenshot%20from%202021-02-05%2001-20-43.png)

입력데이터의 개수를 2로 나누는 것을 xavier initialization에 추가했더니 잘 동작한다.  

적당한 초기화 값을 찾는 연구는 진행중이다. 다양한 방법을 시도해보자!!  

