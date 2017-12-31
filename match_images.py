'''
import cv2
import numpy as np
from scipy import ndimage
import pickle
MIN_MATCH_COUNT=10
frame_rec=7
threshold=5
detector=cv2.ORB(nfeatures=2000);
FLANN_INDEX_LSH=6
flannParam=dict(algorithm=1,trees=4) #2
flann=cv2.FlannBasedMatcher(flannParam,{})
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
trainImg=cv2.imread("chandelier.jpg")
trainImg = cv2.filter2D(trainImg, -1, kernel)
#original = cv2.imread("book.jpg", 0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
#trainKP=pickle.load(open('keypoints_chandelier.p','rb'))
#trainDesc=pickle.load(open('descriptors_chandelier.p','rb'))
#h,w=trainImg.shape
#print len(trainKP)
cam=cv2.VideoCapture(0)
counter=0.0
total=0.0
#cam.set(3,h)
#cam.set(4,w)
#img = cv2.filter2D(img, -1, kernel)
while True:
    total=total+1.0
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    #   QueryImg=cv2.filter2D(QueryImg,-1,kernel)
    #QueryImg=cv2.resize(QueryImg,(h,w))
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,threshold)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
    #print len(matches)
    goodMatch=[]
    try:
        for m,n in matches:
            if(m.distance<0.7*n.distance):
                goodMatch.append(m)
    except:
    	#print "error"
    	r=1
    if(len(goodMatch)>MIN_MATCH_COUNT):
        counter=counter+1.0
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,5.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        
    if total==10 and counter==3:
    	print "object detected- %d" %(counter)
    	total=0
    	counter=0
    elif total==10 and counter<3:
    #else:
        print "Not Enough match found- %d/%d/%d"%(len(goodMatch),MIN_MATCH_COUNT,counter)
        total=0
        counter=0
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
 '''




import cv2
import numpy as np
MIN_MATCH_COUNT=10
FLANN_INDEX_LSH=6
detector=cv2.ORB()

#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
trainImg=cv2.imread("abhishaar.jpg", 0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

cam=cv2.VideoCapture(0)
total=0
counter=0
while True:
    total=total+1.0
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    try:
        for m,n in matches:
            if(m.distance<0.75*n.distance):
                goodMatch.append(m)
    except:
        r=1
    if(len(goodMatch)>MIN_MATCH_COUNT):
        counter=counter+1.0
        #print "object detected"
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,5.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
    if total==10 and counter>=3:
        print "object detected (Abhishaar)"    	
        cv2.putText(QueryImgBGR,'object detected', (50, 50), cv2.FONT_ITALIC, 0.8, 255)
        counter=0
        total =0 
    elif total==10 and counter<3:
        #print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
        cv2.putText(QueryImgBGR,'Not Enough matches found', (50, 50), cv2.FONT_ITALIC, 0.8, 255)
        counter=0
        total=0
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
