# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:33:31 2023

@author: ppjj2
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#한국어 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname = 'C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family = font_name)
plt.rcParams['axes.unicode_minus'] = False

#데이터 불러오기
data1 = pd.read_csv("5대+범죄+발생현황_20231122091636.csv")
data2 = pd.read_csv("서울시 자치구 연도별 방범용 CCTV 운영 현황_230630기준.csv",encoding="cp949",thousands=',')
data3 = pd.read_csv("서울시 유흥주점영업 인허가 정보.csv",encoding="cp949")
data4 = pd.read_csv("주민등록인구(월별)_20231123143545.csv")
data5= pd.read_csv("행정구역(구별)_20231127113657.csv")

#데이터 정제하기
data1 = data1.drop(['자치구별(1)'], axis = 1)
data_area = data1['자치구별(2)'].drop([0,1,2], axis = 0)
data_area = data_area.reset_index(drop = True)

#15년~21년 범죄 데이터 합계만 뽑음
data15 = data1['2015']
data16 = data1['2016']
data17 = data1['2017']
data18 = data1['2018']
data19 = data1['2019']
data20 = data1['2020']
data21 = data1['2021']
data22 = data1['2022']

#데이터 열과 행이 복잡하게 되어있어서 이렇게 각각 하나씩 나눠서 concat 진행
max_criminal = pd.concat([data15,data16,data17,data18,data19,data20,data21,data22], axis = 1)
max_criminal = max_criminal.drop([0,1,2], axis = 0)
max_criminal = max_criminal.reset_index(drop = True)
max_criminal = max_criminal.astype(float)

#index 값에 자치구 이름을 넣기 위해 concat 하고 index 다시 세팅
max_criminal = pd.concat([data_area, max_criminal], axis = 1)
max_criminal.set_index("자치구별(2)", inplace=True)

#8년동안 범죄가 가장 많았던 구 Top5
#강남구, 송파구, 영등포구, 관악구, 강서구
max_criminal_Top5 = max_criminal.sum(axis = 1)
max_criminal_Top5 = max_criminal_Top5.drop('소계')
max_criminal_Top5.sort_values(ascending=False, inplace=True)
max_criminal_Top5 = max_criminal_Top5.iloc[0:5]

#시각화
max_criminal_Top5.plot.bar(rot=0)
plt.ylabel('범죄 현황')
plt.xlabel('지역')
plt.title('8년간 범죄 Top5')
plt.show()

#범죄 최다년도 최소년도 구하기
max_criminal_value = max_criminal.iloc[0,:].idxmax() #2015년에 범죄 최다
min_criminal_value = max_criminal.iloc[0,:].idxmin() #2021년에 범죄 최소
max_criminal_plot = max_criminal.iloc[0,:]

max_criminal_plot.plot.bar(rot=0)
plt.ylabel('합계')
plt.xlabel('년도')
plt.title('범죄 합계')
plt.show()

#2015년도와 2022년도의 Top5 지역별 데이터를 시각화해서 비교
#강남구 송파구 관악구 영등포구 강서구
data_compare = pd.concat([data15, data21, data22], axis = 1)
data_compare.drop([0,1,2,3], axis = 0,inplace=True)
data_area = data1['자치구별(2)'].drop([0,1,2,3], axis = 0)
data_area = data_area.reset_index(drop = True)
data_compare.set_index(data_area, inplace=True)
data_compare = data_compare.astype(float)
data_compare.sort_values(by='2022' ,ascending=False, inplace=True)
data_compare_top5 = data_compare.iloc[0:5,:]


#데이터 시각화 2021년에 확실하게 범죄가 감소
data_compare_top5.plot.bar(rot=0)
plt.ylabel('범죄 현황')
plt.xlabel('지역')
plt.title('2022 범죄 Top5')
plt.show()

#2022년 데이터만 가지고 Top5 시각화
data_compare_top5 = data_compare_top5['2022']
data_compare_top5.plot.bar(rot = 0)
plt.ylabel('범죄 현황')
plt.xlabel('지역')
plt.title('2022 범죄 Top5')
plt.show()

#cctv data 마찬가지로 2015년도와 2022년도 data만 추출
cctv15 = data2['2015년']
cctv22 = data2[['2022년']]


#범죄 Top 5의 cctv data 시각화
cctv_compare = pd.concat([cctv15,cctv22], axis = 1)
cctv_compare.dropna(inplace = True)
cctv_compare = cctv_compare.astype(float)
cctv_compare.insert(0, '지역', data_area)
cctv_compare.sort_values(by = '2022년' , ascending=False, inplace=True)
cctv_compare.head()



#cctv data Top5 새로운 프레임 만들기
cctv_compare_top5 = cctv_compare.loc[cctv_compare['지역'] == '강남구']
cctv_compare_top5 = cctv_compare_top5.append(cctv_compare.loc[cctv_compare['지역'] == '송파구'])
cctv_compare_top5 = cctv_compare_top5.append(cctv_compare.loc[cctv_compare['지역'] == '관악구'])
cctv_compare_top5 = cctv_compare_top5.append(cctv_compare.loc[cctv_compare['지역'] == '영등포구'])
cctv_compare_top5 = cctv_compare_top5.append(cctv_compare.loc[cctv_compare['지역'] == '강서구'])
cctv_compare_top5.set_index('지역', inplace=True)

#cctv 증가율
cctv_compare.set_index('지역', inplace=True)
cctv_increase = (cctv_compare['2022년']/cctv_compare['2015년'])*100
cctv_increase.median()
cctv_increase.mean()


#시각화
cctv_compare_top5.plot.bar(rot=0)
plt.ylabel('cctv 현황')
plt.title('2022년 범죄 Top5 cctv 현황')
plt.show()

#2022년 cctv 많은 Top5 현황
cctv_compare = cctv_compare[['지역', '2022년']]
cctv_compare = cctv_compare.head()
cctv_compare.set_index('지역', inplace=True)
cctv_compare.plot.bar(rot=0)
plt.ylabel('cctv 현황')
plt.xlabel('지역')
plt.title('2022년 cctv Top5 현황')
plt.show()

#추세선으로 비교
#15년~21년 범죄, 검거 합계 데이터만 뽑아서 검거율 계산
#cctv 증가에 따른 검거율 추세 확인
data15 = data1[['2015','2015.1']]
data15 = data15.iloc[3,:].astype(float)
data15 = data15.iloc[1]/data15.iloc[0]
data15 = data15*100

data16 = data1[['2016','2016.1']]
data16 = data16.iloc[3,:].astype(float)
data16 = data16.iloc[1]/data16.iloc[0]
data16 = data16*100

data17 = data1[['2017','2017.1']]
data17 = data17.iloc[3,:].astype(float)
data17 = data17.iloc[1]/data17.iloc[0]
data17 = data17*100

data18 = data1[['2018','2018.1']]
data18 = data18.iloc[3,:].astype(float)
data18 = data18.iloc[1]/data18.iloc[0]
data18 = data18*100

data19 = data1[['2019','2019.1']]
data19 = data19.iloc[3,:].astype(float)
data19 = data19.iloc[1]/data19.iloc[0]
data19 = data19*100

data20 = data1[['2020','2020.1']]
data20 = data20.iloc[3,:].astype(float)
data20 = data20.iloc[1]/data20.iloc[0]
data20 = data20*100

data21 = data1[['2021','2021.1']]
data21 = data21.iloc[3,:].astype(float)
data21 = data21.iloc[1]/data21.iloc[0]
data21 = data21*100

data22 = data1[['2022','2022.1']]
data22 = data22.iloc[3,:].astype(float)
data22 = data22.iloc[1]/data22.iloc[0]
data22 = data22*100

#위 값으로 데이터 프레임 새로 만들기
data = {'Category': ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'],
        'Values': [data15, data16, data17, data18, data19, data20, data21, data22]}
arrest_rate = pd.DataFrame(data)
arrest_rate.drop(['Category'], axis = 1, inplace=True)


cctv_sum = data2.iloc[:,2:10].sum()
cctv_sum = cctv_sum.reset_index(drop = True)

arrest_combined_cctv = pd.concat([arrest_rate, pd.DataFrame(cctv_sum)], axis=1, ignore_index=False)

#추세선 시각화
sns.scatterplot(x=arrest_combined_cctv[0], y=arrest_combined_cctv['Values'])
sns.regplot(x=arrest_combined_cctv[0], y=arrest_combined_cctv['Values'])
plt.xlabel("cctv개수")
plt.ylabel("검거율")
plt.title("cctv수 증가에 따른 검거 추세")
plt.grid()
plt.show() #추세선으로 보아 cctv의 개수가 증가하면서 검거율이 증가하는 것을 알 수 있다.


#인구에 따른 범죄횟수 추세선
#서울시 자치구별 인구수
data4_1 = data4[['2021. 12.1']]
data4_1 = data4_1.drop([0,1,2], axis = 0)
data4_1 = data4_1.reset_index(drop = True)
data4_1 = data4_1.astype(float)
data4_1.insert(0, '지역', data_area)


#서울시 자치구별 범죄발생 횟수
data1_1 = data1[['2022']]
data1_1 = data1_1.drop([0,1,2,3], axis = 0)
data1_1 = data1_1.reset_index(drop = True)
data1_1 = data1_1.astype(float)

data_population_criminal = pd.concat([data4_1,data1_1], axis = 1)

sns.scatterplot(x=data_population_criminal['2021. 12.1'], y=data_population_criminal['2022'])
sns.regplot(x=data_population_criminal['2021. 12.1'], y=data_population_criminal['2022'])
plt.xlabel("인구수")
plt.ylabel("범죄 횟수")
plt.title("인구수에 따른 범죄")
plt.grid()
plt.show()

#cctv 개수에 따른 범죄횟수 추세선
#서울시 자치구별 cctv 수
data2_1 = data2[['2022년']]
data2_1.dropna(inplace = True)

data_cctv_criminal = pd.concat([data2_1,data1_1], axis = 1)

sns.scatterplot(x=data_cctv_criminal['2022년'], y=data_cctv_criminal['2022'])
sns.regplot(x=data_cctv_criminal['2022년'], y=data_cctv_criminal['2022'])
plt.xlabel("cctv수")
plt.ylabel("범죄 횟수")
plt.title("cctv수에 따른 범죄")
plt.grid()
plt.show()

#지역 면적 1km^2 당 cctv의 개수
#면적
data5_1 = data5.iloc[:,1:3]
data5_1 = data5_1.drop([0,1,2], axis = 0)
data5_1 = data5_1.reset_index(drop = True)
#2022년 cctv개수
data2_2 = data2[['2022년']]
data2_2.dropna(inplace = True)
#2022년 범죄횟수
data1_2 = data1[['2022']]
data1_2 = data1_2.drop([0,1,2,3], axis = 0)
data1_2 = data1_2.reset_index(drop = True)
data1_2 = data1_2.astype(float)
data1_2.rename(columns={'2022': '2022 범죄 횟수'}, inplace=True)
#세게 합치기
data_area_cctv = pd.concat([data5_1,data2_2,data1_2], axis = 1)
data_area_cctv.set_index('자치구별(2)', inplace=True)
data_area_cctv = data_area_cctv.astype(float)

#지역 면적 1km^2 당 cctv 개수
data_area_cctv['1km^2당 cctv'] = data_area_cctv['2022년']/data_area_cctv['2022']

print(data_area_cctv['1km^2당 cctv'].mean()) #145.00081509277837
data4_1.set_index('지역', inplace=True) 
data_area_cctv['2022년 인구수'] = data4_1

data_area_cctv_population = data_area_cctv[data_area_cctv['1km^2당 cctv'] > data_area_cctv['1km^2당 cctv'].mean()]


sns.scatterplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['2022 범죄 횟수'])
sns.regplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['2022 범죄 횟수'])
plt.xlabel("1km^2당 cctv")
plt.ylabel("범죄 횟수")
plt.title("1km^2당 cctv개수에 따른 범죄 횟수")
plt.grid()
plt.show()

#지역별 유흥업소 개수
data3_1 = data3[data3['영업상태명'] == '영업/정상']
data3_2 = pd.DataFrame(data3_1['지번주소'].fillna('').apply(lambda x: x.split(' ', 2)).tolist())
count_values = data3_2[1].value_counts()

#인구당 범죄 횟수
population_per = (data_area_cctv['2022 범죄 횟수']/data_area_cctv['2022년 인구수'])*100000
population_per.sort_values(ascending=False, inplace=True)
population_per_top5 = population_per.head()
population_per_top5.plot.bar(rot=0)
plt.xlabel('지역')
plt.ylabel('10만명 당 범죄')
plt.title('10만명 당 범죄 횟수 top5')
plt.show()

#지역별 검거율 -> 의미없는 데이터
arrest = data1[['2022','2022.1']]
arrest = arrest.iloc[4:,:].astype(float)
arrest = arrest['2022.1']/arrest['2022']
arrest = arrest*100
arrest = pd.DataFrame(arrest)
arrest = arrest.reset_index(drop = True)
arrest.insert(0, '지역', data_area)
arrest.set_index('지역', inplace=True)
data_area_cctv['2022년 지역별 검거율'] = arrest

sns.scatterplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['2022년 지역별 검거율'])
sns.regplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['2022년 지역별 검거율'])
plt.xlabel("1km^2당 cctv")
plt.ylabel("2022년 지역별 검거율")
plt.grid()
plt.show()

#종로 값이 너무 커서 이상하게 나온 것 같아서 제외하고 진행해봄
index_to_drop = 0
data_area_cctv = data_area_cctv.drop(data_area_cctv.index[index_to_drop])
data_area_cctv['2022년 지역별 검거율'].median()

sns.scatterplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['2022년 지역별 검거율'])
sns.regplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['2022년 지역별 검거율'])
plt.xlabel("1km^2당 cctv")
plt.ylabel("2022년 지역별 검거율")
plt.grid()
plt.show()

#10만명 당 범죄 수와 1km^2당 cctv의 추세선 -> 의미 없다
data_area_cctv['10만명 당 범죄 수'] = population_per

sns.scatterplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['10만명 당 범죄 수'])
sns.regplot(x=data_area_cctv['1km^2당 cctv'], y=data_area_cctv['10만명 당 범죄 수'])
plt.xlabel("1km^2당 cctv")
plt.ylabel("10만명 당 범죄 수")
plt.grid()
plt.show()

#10만명 당 범죄 수와 지역별 유흥업소 개수 top5 비교 용산구를 제외한 전부 포함
count_values.sort_values(ascending=False, inplace=True)
count_values.head()
population_per.head()

count_values.head().plot.bar(rot=0)
plt.xlabel('지역')
plt.ylabel('유흥업소 개수')
plt.title('지역별 유흥업소 개수 Top5')
plt.show()

#결론 cctv는 검거에는 도움이 되지만 범죄예방에는 큰 도움이 되지않고, 인구수에 비례해서
#범죄가 많이 발생하지만 인구단위를 맞춰서 분석을 진행하면 유흥업소가 많은 곳이 범죄가 
#많이 발생하는 것을 알 수 있다.
#번외로 노원구의 데이터와 강남구의 데이터 보여주고 cctv의 개수 차이가 너무 많이 난다.
#인구가 비슷한 상황 -> 지역 예산이 강남은 13억인데 노원은 1억도 안되는 상황
#지역별로 예산 차이가 큰 상황이다. 예산을 늘려 
#개수뿐만 아니라 사각지대에도 잘 설치해야한다.