# -2022_2_BA-Dimensionaly_Reduction_by_GA
Tutorial Homework 1(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)

1. 튜토리얼의 목적 : Genetic Algorithm을 이용한 다변량 회귀분석 데이터셋의 Dimensionaly reduction 수행 및 결과 제시

2. 데이터셋 : 1978년 보스턴 내 506개 타운의 주택가격 데이터셋(sklearn.datasets, boston)
 - 종속변수 : MEDV(보스턴 내 타운별 주택가격 중앙값)
 - 독립변수 : CRIM(범죄율), INDUS(비소매상업지역 면적 비율), NOX(일산화질소 농도), RM(주택당 방 수), LSTAT(인구 중 하위계층 비율), B(인구 중 흑인 비율), PTRATIO(학생/교사 비율), ZN(25,000ft^2를 초과하는 거주지역 비율), CHAS(찰스 강 경계 위치 여부), AGE(1940 전 건축된 주택 비율), RAD(고속도로까지의 거리), DIS(직업센터까지의 거리), TAX(재산세율)
 - boston.json으로 저장

3. 코드 설명([BA]01_dimensionality_reduction(GA).ipynb, 01_dimensionality_reduction.py)

 (3-1) 코드 개요 : Genetic Algorithm을 이용한 Dimensionaly reduction
 - Hyper parameter 목록 : MUT(변이확률, %로 표현 : 변이확률 5%는 MUTE = 0.05가 아닌 MUTE = 5로 입력), END(iteration 횟수, iteration 횟수를 채우면 알고리즘 종료), popCOUNT(해집단 내 Chromosome 개수), selCOUNT(총 selection 개수), numCOL(Chromosome 1개 당 유전자 개수)
 - 코드 구성

 (3-2) 분석 준비
  - pandas 등 기초적인 분석 패키지와 머신러닝 패키지 sklearn import
  - sklearn에서 boston dataset loading(X, Y concat 한 후, .json 파일 형태로 저장함)

 (3-3) Initialization
  - hyperparameter 설정 : 변이확률 5%, iteration 횟수 200회, population size 50, selection size 25
  - 초기 해 설정

 (3-4) Fitness Evaluation
  - 사용된 fitness fuction : 다중회귀모형의 Root Mean Square Error(RMSE)
  - chromosome별로 설정된 해에 따라 사용되는 독립변수 구분
  - 구분된 독립변수에 따라 종속변수에 대한 다중회귀모형 구축
  - 구축된 다중회귀모형의 RMSE를 GA 성능지표로 활용(낮을수록 우수한 성능)

 (3-5) Select the superiors
  - chromosome 내 각 gene에 대하여 다음과 같이 가중치 도출 : Max(gene) - gene (낮을수록 좋은 성능지표의 특성 반영)
  - 가중치에 따라 다음 세대에 유전자를 전달할 chromosome selection

 (3-6) Crossover
  - 1-crossover point 방식을 이용한 상속
  - 설계된 상속에 따라 crossover 연산자 구축

 (3-7) Mutation
  - 설정된 Mutate rate(5%)에 따라 일부 유전자 변환(0 -> 1 or 1 -> 0)

 (3-8) Result
  - 앞서 정의된 모든 과정을 함수로 정의
  - loop 문을 이용하여 END까지 iteration 수행, END 도달 후 알고리즘 종료

4. 결과 분석

 (4-1) 성능지표 변화
 
![image](https://user-images.githubusercontent.com/106015570/195500049-f7aba6e0-232c-47a8-b002-5a906fb2e53a.png)

  - 5회 정도의 iteration으로 변곡점 도달
  - Mutate를 거치면서 5.3573에서 5.3569로 성능지표 미량 변화

 (4-2) 최종 chromosome

[[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 5.356858545742943]

   - 선택된 변수 : RM, LSTAT, B, PTRATIO, ZN, CHAS, AGE, RAD, DIS, TAX
   - GA를 이용하여 13개 변수를 10개로 축소
   - 다만 차원 축소 측면에서 많은 양이 축소되지는 않음 
   - 차원을 더 많이 축소하기 위해서는, GA와 같은 선택 방법보다는 PCA, MDS 등 추출 방법을 쓰는 것이 효과적일 것으로 보임 
