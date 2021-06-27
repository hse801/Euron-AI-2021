#### 📚 Stanford cs231n  
#### 🚩 Euron AI team 황시은 9주차 예습  
### CNN Architectures

- LeNet
	- ConvNet을 최초로 도입  
	- 우편번호, 숫자 등에 사용됨  
- AlexNet
![AlexNet](https://cheong.netlify.app/static/b51f4f8adf20720c66d5775df980adf3/799d3/image4.png)

	- ReLU 첫 도입  
	- dropout 도입  

- VGG  
![vggnet](https://cheong.netlify.app/static/997ebd9a509c3d4dabe174e8d1b35bc8/799d3/image9.png)

	- 더 깊은 네트워크, 작은 필터 사용

- GoogLeNet
	- 22개의 layer  
	- 깊지만 효율은 증대
	- inception module 사용  
	- FC layer 삭제  

- ResNet
![resnet architecture](https://cheong.netlify.app/static/57915acac15aa5579816d261127c7158/799d3/image23.png)

	- Very deep networks using residual connections  
	- residual block들을 쌓아올려 구성  

- SENet(Squeeze-and-Excitation Networks)
	- adaptive feature map reweighting  이용  
	- feature recalibration 적용  


- Wide Residual Networks
![wide residual netrowk](https://cheong.netlify.app/static/b7d551d0c489f3e7d57d83f3acad2f61/799d3/image28.png)


	- 발전된 ResNet  
	- depth가 아니라 residual이 중요하다
	- depth보다 width를 늘리는 것이 계산 상 효율적이다  

- ResNeXt
	- 발전된 ResNet
	- pathway를 이용해 width를 늘리는 방식 이용  

- FractalNet
	- fractal 구조로 네트워크 구성  
	- residual representation 필요 없고 전달 잘 하는 것이 중요  

- Densely Connected Convolutional Networks  
	- vanishing gradient 약호
	- feature의 전파가 강화됨
	- feature가 더 잘 재사용됨  

- MobileNets
	- 더욱 가볍고 효율적인 신경망  
	- loss의 손실이 적으면서 효율적  

- Meta-learning
	- 네트워크의 구조 자체를 학습하는 것  




