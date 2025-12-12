# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장 가능

'''
히스토그램 : 도수분포표를 그래프로 나타낸 것이다. 
    변수가 하나인 단변수 데이터의 빈도수를 표현하다. x축을 같은 크기의
    여러구간으로 나누고 각 구간에 속하는 데이터값의 갯수를 y축에 표시한다.
도수분포표 : 도수 분포는 표본의 다양한 산출 분포를 보여주는 표이다. 
    가령 성인 30명을 대상으로 하루동안 사용하는 문자의 건수를 조사하여
    10~20건, 20~30건에 각 몇명이 분포하는지를 표시한다. 
커널밀도그래프 : 주어진 데이터를 정규화시켜 넓이가 1이 되도록 그린 그래프.
'''

# 인자 받기 (이미지 저장 경로)
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
else:
    output_dir = os.getcwd()  # 현재 디렉토리

#타이타닉 데이터셋을 로드한다. 
titanic = sns.load_dataset('titanic')
sns.set_style('darkgrid')

#Axe객체를 생성하여 3개의 영역으로 나눈다. 수평방향으로 그래프가 표시된다.
fig = plt.figure(figsize=(15, 5))
axe1 = fig.add_subplot(1,3,1)
axe2 = fig.add_subplot(1,3,2)
axe3 = fig.add_subplot(1,3,3)

#히스토그램+커널밀도그래프 (distplot 대체)
sns.histplot(titanic['fare'], kde=True, ax=axe1)
#커널밀도그래프
sns.kdeplot(x='fare', data=titanic, ax=axe2)
#히스토그램
sns.histplot(x='fare', data=titanic, ax=axe3)

#타이틀 설정
axe1.set_title('titanic fare - hist/kde')
axe2.set_title('titanic fare - kde')
axe3.set_title('titanic fare - hist')

# 이미지 파일 저장
filename = "titanic_histogram.png"
output_path = os.path.join(output_dir, filename)
plt.savefig(output_path, dpi=100, bbox_inches='tight')
plt.close()  # 메모리 정리

print(f"타이타닉 fare 히스토그램 3개 그래프 저장 완료!")
print(f"이미지 경로: {output_path}")
