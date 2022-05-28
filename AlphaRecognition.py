import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import OS,ssl,time

#Fetching the data
X=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M',"N","O","P","Q","R","S","T",'U','V','W',"X","Y","z"]
nclasses = len(classes)

if(not os.environ.get('PYTHONHTTPVERIFY','')and
    getattr(ssl,'_create_unveriefied_context',None)):
    ssl._create_default_https_context=ssl._create_unverified_context

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 0,train_size = 700,test_size = 600)
Xtrain_scaled = X_train/255.0
Xtest_scaled = X_test/255.0
clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(Xtrain_scaled,y_train)

ypred = clf.predict(Xtest_scaled)
accuracy = accuracy_score(y_test,ypred)
cap=cV2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        grey = cV2.cvtColor(frame,cV2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cV2.rectangle(grey,upper_left,bottom_right,(0,255,0),2)
        roi = grey[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        im_pil = Image.fromarray(roi)
        image_bw  =im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resize)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scale = np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_resized_inverted)

        image_bw_resized_inverted_scale = np.asarray(image_bw_resized_inverted_scale)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scale).reshape(1,784)
        test_predict = clf.predict(test_sample)
        print("predicted class",test_predict)

        cV2.imshow('frame',grey)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e :
        pass
cap.release()
cv2.destroyAllWindows()


