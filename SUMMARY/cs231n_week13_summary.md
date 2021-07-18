#### 📚 Stanford cs231n  
#### 🚩 Euron AI team 황시은 13주차 예습  
### Lecture 13 | Generative Models  

- Supervised vs Unsupervised Learning  
	- Supervised learning  
			- Data : (x, y)  
			- To learn a function to map x-> y 
			- Classification, regression, segmentation, etc  
	- Unsupervised Learning
			- Just data, no labels
			- Learn some underlying hidden structure of the data
			- Ex) Clustering, feature learning(auto encoders), etc  

- Generative Models  
![](https://www.dropbox.com/s/l3xa29mk7nl8jtp/Screenshot%202018-06-10%2010.20.02.png?raw=1)

	- Train data와 동일한 분포의 새로운 샘플들을 생성한다
	- 데이터로부터 실제와 같은 샘플을 얻을 수 있다  
	- 일반적인 특징을 찾을 때 유용한 latent representation을 추론할 수 있음  
	- High dimensional prob, distribution을 추출해서 다룰 수 있음  

![](https://www.dropbox.com/s/2pzxib21t7z6x9k/Screenshot%202018-06-10%2010.27.26.png?raw=1)

- Fully visible belief network
![image](https://user-images.githubusercontent.com/61612117/126066907-676c3825-1afc-45af-b1e5-13cc975096a7.png)


- PixelRNN  
![](https://www.dropbox.com/s/brrhj1ag50znpkp/Screenshot%202018-06-10%2010.48.40.png?raw=1)

	- 코너부터 시작해 이미지 픽셀 생성  

- PixelCNN  
![](https://www.dropbox.com/s/n3pflqtf64vr99g/Screenshot%202018-06-10%2010.51.17.png?raw=1)
		- 코너부터 이미지 픽셀 생성  
		- CNN over context region을 사용해 이전 픽셀에 dependency  



### Generative Adversarial Networks(GAN)  

- 복잡하고 고차원의 샘플을 원하는데 이를 직접적으로 할 방법이 없으니까 간단한 분보의 샘플을 얻고 그로부터 변형을 학습  
- Traing  GANs: Two-Player game
  ![](https://www.dropbox.com/s/2fkftsb2sksw8ue/Screenshot%202018-06-11%2021.33.15.png?raw=1)

	-	Generator network: 실제와 같은 이미지를 생성해 discriminator를 속인다  
	-	Discriminator network: 실제와 가짜 이미지를 구별한다  
		
		
