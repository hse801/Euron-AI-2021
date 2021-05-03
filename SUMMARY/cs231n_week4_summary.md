#### 📚 Stanford cs231n  
#### 🚩 Euron AI team 황시은 3주차 예습  
# Introduction to Neural Networks  


![image](https://user-images.githubusercontent.com/61612117/116815189-6520ec80-ab97-11eb-8d24-bf3dc44bc685.png)
- Pros  
	- Back Propagation 할 때 사용 가능  
	- Complex Function에 유용 -> CNN, Neural Turing Machine  
	- Chain Rule을 재귀적으로 사용 -> 모든 변수에 대한 gradient 계산하기 위해  

예제  

![image](https://user-images.githubusercontent.com/61612117/116815240-9d282f80-ab97-11eb-9cb8-2e743bafc48e.png)

- Backpropagation Chain-Rule  
	- x, y를 입력으로 받아 z를 출력하는 함수 x가 있다고 할 때 f = (x + y)z로 정의할 수 있다.   
	- 이때, x에 대한 L의 기울기, y에 대한 L의 기울기를 chain-rule을 통해 구할 수 있다.   
	- 각각의 변수에 대해 기울기(편미분값)을 구하기 위해서 Chain-rule 사용  
	- Upstream Gradient, Local Gradient  
		-> 이 두 gradient를 곱해서 x, y의 L에 대한 gradient를 구할 수 있다. 
	- 이 방법이 **Chain-Rule을 이용한 오차역전파(Backpropagation)**이다.  

![image](https://user-images.githubusercontent.com/61612117/116815496-cb5a3f00-ab98-11eb-9b30-bfb4f275acf8.png)
- Local Gradient: 우리가 구하려던 것은 아님  
- Gradients: loss 대비 우리가 구하려던 것  
- 각각의 노드는 주변 환경을 알고 있다.(Immediate surrounding)  
- 국소적 계산(Local computation): 전체에서 어떤 일이 벌어지든 상관 없이, 자신과 관계된 정보만으로 결과를 출력 가능하다. 즉 복잡한 계산도 나누면 단순한 계산이 가능함  
- 위의 구조가 하나의 노드에서 일어나는 연산을 그림으로 나타낸 것이다. 이렇게 구해진 gradient(global gradient)를 뒤로 다시 보내고, 뒤에 있는 local gradient와 받은 global gradient를 곱해서 다시 구하는 과정을 반복하여 가장 적합한 gradient를 찾아간다.   
![image](https://user-images.githubusercontent.com/61612117/116844039-d48cef80-ac1c-11eb-88e2-e55cdb780cd1.png)
![image](https://user-images.githubusercontent.com/61612117/116844062-e8385600-ac1c-11eb-8017-0f4d97f13a88.png)

![image](https://user-images.githubusercontent.com/61612117/116844102-0a31d880-ac1d-11eb-8665-7b0ffe485635.png)
Local gradient를 구하여 sigmoid gate를 더 쉽게 계산할 수 있다. 

![image](https://user-images.githubusercontent.com/61612117/116853655-643c9900-ac31-11eb-95cc-582050bc1315.png)
Activation Function은 특정 무언가를 써야한다기보다는 자신의 모델에 맞게 다양하게 적용해보고 선택한다.
![image](https://user-images.githubusercontent.com/61612117/116853758-8c2bfc80-ac31-11eb-81fa-1fd6110395ee.png)
Forward Propagation: Loss를 구하는 과정
Back Propagation: 반대로 loss에서 input 쪽으로 가며 gradient를 구하는 과정
![image](https://user-images.githubusercontent.com/61612117/116853868-b2ea3300-ac31-11eb-9cbd-c3c1f942ec61.png)
