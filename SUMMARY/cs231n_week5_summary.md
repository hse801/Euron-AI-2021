#### 📚 Stanford cs231n  
#### 🚩 Euron AI team 황시은 5주차 예습  
### Convolutional Neural Networks  

#### 1.  History of Convolutional Neural Networks  
- Mark 1 Perceptron  
- Chain Rule을 이용한 back propagation 발견  
- AlexNet  
- CNN 등장 -> 우체국에서 번호 인식  
#### 2. Convolutional Neural Networks  
- Fully Connected Layer  
![image](https://user-images.githubusercontent.com/61612117/117527779-7c356380-b009-11eb-9d5d-663f6f8dad25.png)
	- 학습한 데이터를 기반으로 분류를 하는 Layer  
	- 보통 CNN의 마지막에 Output 출력을 위해 활용된다  

- Convolutional Layer
![image](https://user-images.githubusercontent.com/61612117/117527844-de8e6400-b009-11eb-9160-3f3f203b4313.png)  
	- 입력된 이미지에서 테두리, 선, 색 등 이미지의 특징을 감지하기 위한 Layer  
	-  각 이미지는 filter와의 연산(내적, dot product)를 통해 새로운 activation map을 생성함  
	- 32x32는 이미지의 크기, 3은 depth, 즉 인풋 이미지에서 RGB 값을 말한다. (depth가 1이면 grayscale)  
	- filter의 depth는 항상 인풋 이미지의 depth와 동일해야 함  
	- activation map의 크기, 개수는 stride, padding, filter의 값에 따라 달라짐  

- Convolution  연산 과정  
![image](https://user-images.githubusercontent.com/61612117/117527971-c834d800-b00a-11eb-8f8a-f51ced3489fd.png)

	- convolution 연산은 필터를 stride만큼 slide해가며 필터와 이미지의 픽셀이 겹치는 것들의 곱의 합으로 계산한다.   
	- output size는 항상 양의 정수  
	- padding은 이미지 주변을 숫자들로 채우는 것
	- padding은 output size를 input size와 동일하게 하기 위해 활용된다
	- 보통 zero padding
	- padding을 사용하지 않는다면 layer를 지날 때마다 이미지의 크기가 작아질 것이다.   

- Pooling Layer
![image](https://user-images.githubusercontent.com/61612117/117528036-2bbf0580-b00b-11eb-99d9-36f973b17cf6.png)
![image](https://user-images.githubusercontent.com/61612117/117528076-5610c300-b00b-11eb-95d3-bbc203d233fc.png)

	- 이미지를 downsampling, 즉 크기를 축소시킨다  
	- 해당 필터 안에서 최댓값을 선택한다. 탐지된 특징이 보존됨  


![](https://blog.kakaocdn.net/dn/qskZw/btqNyabsKQV/qg12NyFpSDKOeBS67Oi0Q0/img.png)

