import os
from os import listdir
from os.path import isfile,isdir, join
import numpy as np
import cv2
import time
import sys




def alphaBlend(img1, img2, mask):
    # print(mask)
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    # print('alpha: ', alpha)
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

def drawRectangle(blurimg, rectangles):
    img = blurimg
    for rect in rectangles:
        img = cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255,255,0), 2)
    return img

def blurFile(face_cascade, license_cascade, inputfile, outputfile, twicefile):
    img = cv2.imread(inputfile)
    print(img.shape)
    scale = 2 # percent of original size
    uniwidth = img.shape[1]
    uniheight = img.shape[0]
    width = int(uniwidth * scale)
    height = int(uniheight * scale)
    dim = (width, height)
    # resize image
    if scale % 2==0:
        resizescale = scale+1
    else:
        resizescale = scale
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    dst = cv2.GaussianBlur(resized,(resizescale,resizescale),cv2.BORDER_DEFAULT)
    print(uniwidth, uniheight)
    circles = []
    rectangles = []
    cv2.imwrite(twicefile, dst)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # print(gray)
    faces = face_cascade.detectMultiScale(gray, 1.2, 1)

    for (x,y,w,h) in faces:
        radius = h
        if w > h:
            radius = w
        
        circles.append([ int((x+w/2)/scale), int((y+h/2)/scale), int(radius/scale)])            
        rectangles.append([int(x/scale), int(y/scale), int((x+w)/scale), int((y+h)/scale)])

    # licenses = license_cascade.detectMultiScale(gray, 1.2, 2)

    # for (x,y,w,h) in licenses:
    #     radius = h
    #     if w > h:
    #         radius = w
        
    #     circles.append([ int((x+w/2)/scale), int((y+h/2)/scale), int(radius/scale)])            
    #     rectangles.append([int(x/scale), int(y/scale), int((x+w)/scale), int((y+h)/scale)])

    # for u in range(0, width, uniwidth):
    #     for v in range(0, height, uniheight):
    #         u1 = (u+uniwidth+int(uniwidth/10))
    #         v1 = (v+uniheight+int(uniheight/10))
    #         if u1 > width:
    #             u1 = width
    #         if v1 > height:
    #             v1 = height
    #         uniimag = dst[v:v1, u:u1]
    #         print(uniimag.shape)
    #         gray = cv2.cvtColor(uniimag, cv2.COLOR_BGR2GRAY)
        
    #         # print(gray)
    #         faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    #         for (x,y,w,h) in faces:
    #             radius = h
    #             if w > h:
    #                 radius = w
                
    #             circles.append([int((x+w/2)/scale), int((y+h/2)/scale), int(radius/9)])            
    blurimg = blurImageWithCircle(img, circles)
    
    # reimg = drawRectangle(blurimg, rectangles)
    # resized = cv2.resize(dst, (img.shape[1]*2, img.shape[0]*2), interpolation = cv2.INTER_AREA)
    


    # filename = 'aresult.jpg'
    cv2.imwrite(outputfile, blurimg)
 
 
def main(argv):
    # face_cascade = cv2.CascadeClassifier('models/haarcascade_fullbody.xml')
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
    license_cascade = cv2.CascadeClassifier('models/license.xml')    
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
        twicefile = outputdir + '/' + 'twice_' + inputfile
        blurFile(face_cascade, license_cascade, infile,outfile, twicefile)
        
        
    # testin = 'testimages/Untitled.jpg'
    # testout = 'output.jpg'
    # blurFile(face_cascade, testin,testout)        
 
if __name__ == "__main__":
   main(sys.argv[1:])