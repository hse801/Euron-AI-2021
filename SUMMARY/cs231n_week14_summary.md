#### 🚩 Euron AI team 황시은 14주차 예습  
### Lecture 14 | Deep Reinforcement Learning  

 ### Reinforcement Learning  
 강화학습: agent가 어떤 행동에 따른 적절한 보상을 통해 보상을 최대화할 수 있는 행동이 무엇인지 학습하는 것  
 ![](https://blog.kakaocdn.net/dn/cVG2bL/btqTV9hICbR/rOKnVf7rz2tzKSG5zQcCe1/img.png) 
Environment와 Agent 간의 상호작용에서 학습 이루어짐  
Agent는 초기 상태에서 특정 Action을 하면, reward와 다음 state를 얻게 되고, 다음 action을 하게 됨  

### Markov Decision Process

Markov Property: 이전 state 와 상관 없이, 과거와 미래 state 는 현재 state 와 완전히 independent 하고, 현재 state 에서 다음 state 로 갈 확률은 항상 같다 라는 성질.
-   state
    -   agent 가 관찰 가능한 상태의 집합.
    -   예) 2차원 grid world 라면, 가능한 모든 (x, y) 좌표
-   action
    -   agent 가 특정 state 에서 행동할 수 있는 action의 집합
    -   예) 2차원 grid world 라면, 상 하 좌 우 이동
-   reward
    -   (state, action) pair 에 따라 env 가 agent 에게 주는 유일한 정보
-   state transition probability
    -   (state, action) pair 에 의해 agent 가 특정 state 로 변경 되야 했지만, env 에 의해 다른 state 로 변경될 확률
-   discount factor
    -   agent 가 받는 reward 중, 현재에 가까운 reward 를 더 비싸게, 현재에 먼 reward 를 더 싸게 해주는 factor.
    -   당장 현재에 있는 reward 가 더 비싸다  

![](https://blog.kakaocdn.net/dn/blysf8/btqTJTU8Yae/EeUHcucwlBKMbjeRVDAMo0/img.png)

### Value function and Q-Value function  
![](https://blog.kakaocdn.net/dn/EaURk/btqT0svevr6/lbELj5uqM2t6qurWXEa6D0/img.png)

### Atari Games  
게임을 학습시켜 높은 점수를 따도록 할 수 있다.  
![](https://t1.daumcdn.net/cfile/tistory/995AF2475C8F97F31F)

### RAM: Recurrent Attention Model  
![](https://t1.daumcdn.net/cfile/tistory/998456435C8F97F124)
입력 이미지가 들어가면 glimpse를 추출한다. NN을 통과시키고, 지금까지 있던 glimpse를 전부 결합시킨다. 출력은 x-y 좌표이다. 이 행동 분포로부터 특정 x, y 위치를 샘플링 한 후에 이 좌표를 통해 다음 glimpse를 얻어낸다. RNN 모델은 다음 위치의 glimpse에 대한 분포를 추출하고 이를 반복한다.   
분류가 목적이므로 마지막에 softmax를 통해 클래스의 확률분포를 출력한다.  


![](https://t1.daumcdn.net/cfile/tistory/99F08D415C8F97F120)


