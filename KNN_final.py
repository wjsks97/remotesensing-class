import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.spatial import distance
os.chdir("C:/Users/MS Byun/Desktop")
fname_img = 'test_pic.jpg'
img = rasterio.open(fname_img)
I = img.read()
I = np.swapaxes(I, 0, 2)
ind=np.shape(I)[0]*np.shape(I)[1]
dim=np.shape(I)[2]
k=10
dat=[[79,645],[287,622],[484,603],[182,558],[386,539],[66,493],[281,483],#water
[320,712],[428,720],[119,419],[208,415],[285,411],[396,422],[481,417],[100,306],[282,296],[459,297],#vegetation
[39,187],[125,204],[280,224],[515,246],[324,357],#building
[86,39],[205,89],[432,32],[477,115],[534,76]]#sky
undef_rgb= I.reshape((ind,dim))
label_rgb=np.empty((10800,3))
label_ind=np.empty((10800,2))
label_group=np.empty((10800,1))
group=np.empty((ind,k))
# labwg=np.empty((10800,4))
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
            label_ind[e]=((dat[i][0]+j)*744+(dat[i][1]+q),g)
            label_group[e]=g
            e=e+1
label_ind=label_ind.astype(int)

dis=distance.cdist(undef_rgb,label_rgb,'euclidean')
index_sort=np.argsort(dis)[:,:k]
for p in range(ind):
    for q in range(k):
        group[p,q]=label_group[index_sort[p,q]]
group=group.astype(int)
      

def defineGroup(arr, ng):
    # x=np.zeros((np.shape(arr)[0],ng+1)).astype(int)
    y=np.zeros((np.shape(arr)[0],1)).astype(int)
    for i in range(np.shape(arr)[0]):
        a=np.bincount(arr[i,:]).astype(int)
        y[i]=np.argmax(a)
    return y
undef2def=defineGroup(group,4)
undef2def[label_ind[i,0]]=label_ind[i,1]
undef2def=undef2def.reshape((558,744,1))
undef2def = np.swapaxes(undef2def, 0, 1)

palette = np.array([[  0,   0, 255],   # blue(water))
                    [  0, 255,   0],   # green(vegetation)
                    [255,   0,   0],   # red(building)
                    [0, 204, 255]])  # skyblue(sky)
pic=np.zeros((744,558,3))
for a in range(np.shape(undef2def)[0]):
    for b in range(np.shape(undef2def)[1]):
        if undef2def[a,b]==1:
            pic[a,b]=palette[0]
        elif undef2def[a,b]==2:
            pic[a,b]=palette[1]
        elif undef2def[a,b]==3:
            pic[a,b]=palette[2]
        else:
            pic[a,b]=palette[3]
            
            
            
plt.figure(figsize=(30,20))
plt.rc('font', size=30)
plt.title('KNN RESULT')
plt.imshow(pic)
