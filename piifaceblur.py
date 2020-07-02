import os
from os import listdir
from os.path import isfile,isdir, join
import numpy as np
import cv2
import time
import sys




def alphaBlend(img1, img2, mask):
    print(mask)
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended




def blurImageWithCircle(originalimg, circles):
    img = originalimg
    H,W = img.shape[:2]
    for circle in circles:
        mask = np.zeros((H,W), np.uint8)
        cv2.circle(mask, (circle[0], circle[1]), circle[2], (255,255,255), -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (21,21),11 )

        blured = cv2.GaussianBlur(img, (21,21), 11)

        img = alphaBlend(img, blured, mask)
    return img


def blurFile(face_cascade, inputfile, outputfile):
    img = cv2.imread(inputfile)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)
    faces = face_cascade.detectMultiScale(gray, 1.2, 1)

    circles = []
    for (x,y,w,h) in faces:
        radius = h
        if w > h:
            radius = w
        
        circles.append([x+int(w/2), int(y+h/2) , radius])            
    blurimg = blurImageWithCircle(img, circles)


    filename = 'aresult.jpg'
    cv2.imwrite(outputfile, blurimg)
 
 
def main(argv):
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
    inputdir = argv[0]
    outputdir = argv[1]
    if isdir(outputdir):
        pass
    else:
        os.mkdir(outputdir)
    inputfiles = [f for f in listdir(inputdir) if isfile(join(inputdir, f))]
    # print(inputdir, outputdir, inputfiles)
    for inputfile in inputfiles:
        infile = inputdir + '/' + inputfile
        outfile = outputdir + '/' + 'blur_' + inputfile
        blurFile(face_cascade, infile,outfile)
        
        
    # testin = 'testimages/Untitled.jpg'
    # testout = 'output.jpg'
    # blurFile(face_cascade, testin,testout)        
 
if __name__ == "__main__":
   main(sys.argv[1:])