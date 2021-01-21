import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas

##  將辨識出的車牌  藉由yolo座標  擷取下來  儲存成小張車牌影像  準備進行透視投影變換

for i in range(1,8):
 img = cv2.imread('truck_or_img/%i.jpg'%i) ## 1開始 i
 #cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
 #cv2.imshow("ori",img)
 #cv2.waitKey(0)


 plate_xy_cor = np.loadtxt('plate_xy_cor.csv',delimiter = ',')

#print(plate_xy_cor.shape)

 #cut_region = plate_xy_cor[5]  ##   0開始 i-1 
 cut_region = plate_xy_cor[i-1]  ##   0開始 i-1 
 cut_region = cut_region.astype('int32')
#print(cut_region)


#after_cut_img = img[cut_region[0]:cut_region[1],cut_region[2]:cut_region[3]]
 after_cut_img = img[cut_region[2]:cut_region[3],cut_region[0]:cut_region[1]]
#cv2.imwrite('cut_plate_img/c6.jpg',after_cut_img)  ## 1開始  i

 cv2.imwrite('t/c%i.jpg'%i,after_cut_img)  ## 1開始  i

 #cv2.namedWindow('after_cut', cv2.WINDOW_NORMAL)
 #cv2.imshow("after_cut",after_cut_img)
 #cv2.waitKey(0)



