import cv2
import torch
import numpy as np
from PIL import Image
# 加载单张图片
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))

site = np.array([[69],[109],[106],[113],[77],[142],[73],[152],[108],[154]])

points = list([(site[i:i+2][0].item(),site[i:i+2][1].item()) for i in range(0,len(site),2)])
print(site)
cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    for point in points:
        cv2.circle(frame, point, 1,(0,0,255), 4)
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
