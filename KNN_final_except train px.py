#KNN구현에 필요한 환경을 위한 모듈 설치 및 import
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.spatial import distance
import pandas as pd

#KNN 알고리즘을 적용할 이미지를 불러와 행렬의 형태로 'I' 변수에 저장
os.chdir("C:/Users/MS Byun/Desktop")
fname_img = 'test_pic.jpg'
img = rasterio.open(fname_img)
I = img.read()
I = np.swapaxes(I, 0, 2)

#이미지에서 KNN 알고리즘 적용 전 training set을 설정. 총 4개의 분류 클래스를 설정했으며 training 시킬 영역의 중앙 픽셀정보를 'dat' 변수에 저장
dat=[[79,645],[287,622],[484,603],[182,558],[386,539],[66,493],[281,483],#water
[320,712],[428,720],[119,419],[208,415],[285,411],[396,422],[481,417],[100,306],[282,296],[459,297],#vegetation
[39,187],[125,204],[280,224],[515,246],[324,357],#building
[86,39],[205,89],[432,32],[477,115],[534,76]]#sky

#KNN 알고리즘 구현에 필요한 K값 및 labeled px, test px, undefined px 정보가 들어갈 행렬 생성
k=10
ind=np.shape(I)[0]*np.shape(I)[1]
dim=np.shape(I)[2]
undef_rgb= I.reshape((ind,dim))
label_rgb=np.empty((10800,3))
label_ind=np.empty((10800,2))
label_group=np.empty((10800,1))
test_water=(pd.read_csv('test_water.csv')).to_numpy()
test_vegetation=(pd.read_csv('test_vegetation.csv')).to_numpy()
test_building=(pd.read_csv('test_building.csv')).to_numpy()
test_sky=(pd.read_csv('test_sky.csv')).to_numpy()
group=np.empty((ind,k))


#이미지 행렬 I로부터 rgb 및 인덱스, class 정보를 미리 만들어놓은 행렬에 저장
e=0
for i in range(27):
    if i<=6:
        g=1
    elif 6<i<=16:
        g=2
    elif 16<i<=21:
        g=3
    elif 21<i:
        g=4
    for j in range(-9,11):
        for q in range(-9,11):
            label_rgb[e,:]=I[dat[i][0]+j,dat[i][1]+q,:]
            label_ind[e]=((dat[i][0]+j)*744+(dat[i][1]+q),0)
            label_group[e]=g
            e=e+1
label_ind=label_ind.astype(int)

#라벨링 픽셀과 라벨링 되지 않은 픽셀간의 RGB 거리 행렬 생성 후 크기순으로 정렬하여 k(=10)개 까지 잘라 group 행렬에 저장
dis=distance.cdist(undef_rgb,label_rgb,'euclidean')
index_sort=np.argsort(dis)[:,:k]
for p in range(ind):
    for q in range(k):
        group[p,q]=label_group[index_sort[p,q]]
group=group.astype(int)
      
#group 행렬로부터 가장 많은 class를 세고 산출하는 함수 'defineGroup' 작성
def defineGroup(arr, ng):
    # x=np.zeros((np.shape(arr)[0],ng+1)).astype(int)
    y=np.zeros((np.shape(arr)[0],1)).astype(int)
    for i in range(np.shape(arr)[0]):
        a=np.bincount(arr[i,:]).astype(int)
        y[i]=np.argmax(a)
    return y

#'defineGroup' 함수를 통해 픽셀의 class 정의 후 training한 영역은 검은색으로 표현하기 위해 0 값 부여
undef2def=defineGroup(group,4)
for i in range(10800):
    undef2def[label_ind[i,0]]=label_ind[i,1]
    
#혼동행렬 작성을 위한 test set 에 대한 분류 결과를 pridict_ 행렬에 저장 후 class별 분류 개수 count
pridict_water=np.empty((1,100))
pridict_veg=np.empty((1,100))
pridict_build=np.empty((1,100))
pridict_sky=np.empty((1,100))
for i in range(100):
    pridict_water[0,i]=undef2def[test_water[i,0]]
    pridict_veg[0,i]=undef2def[test_vegetation[i,0]]
    pridict_build[0,i]=undef2def[test_building[i,0]]
    pridict_sky[0,i]=undef2def[test_sky[i,0]]

pridict_water=pridict_water.astype(int)
pridict_water=np.bincount(pridict_water[0,:])
pridict_veg=pridict_veg.astype(int)
pridict_veg=np.bincount(pridict_veg[0,:])
pridict_build=pridict_build.astype(int)
pridict_build=np.bincount(pridict_build[0,:])
pridict_sky=pridict_sky.astype(int)
pridict_sky=np.bincount(pridict_sky[0,:])


#1차원으로 펴놨던 행렬을 다시 2차원으로 reshape 후 색을 입혀 plot
undef2def2=undef2def.reshape((558,744,1))
undef2def2 = np.swapaxes(undef2def2, 0, 1)

palette = np.array([[  0,   0, 255],   # blue(water))
                    [  0, 255,   0],   # green(vegetation)
                    [255,   0,   0],   # red(building)
                    [0, 204, 255],[0,0,0]])  # skyblue(sky)
pic=np.zeros((744,558,3))
for a in range(np.shape(undef2def2)[0]):
    for b in range(np.shape(undef2def2)[1]):
        if undef2def2[a,b]==1:
            pic[a,b]=palette[0]
        elif undef2def2[a,b]==2:
            pic[a,b]=palette[1]
        elif undef2def2[a,b]==3:
            pic[a,b]=palette[2]
        elif undef2def2[a,b]==0:
            pic[a,b]=palette[4]
        else:
            pic[a,b]=palette[3]
            
plt.figure(figsize=(30,20))
plt.rc('font', size=30)
plt.title('KNN RESULT')
plt.imshow(pic)