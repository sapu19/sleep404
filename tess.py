import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import re
from datetime import datetime


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Image", gray)

# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
filename = open("text.txt","w") 
filename.write(text)
filename.close()


filename = ("text.txt")
with open(filename,"r", encoding="utf-8") as file:
	filedata = file.readlines()
data_cleaned = ""
for line_data in filedata:
	data_cleaned += line_data.replace('\n',' ')
re1 = r'[\d]{1,2}/[\d]{1,2}/[\d]{4}'
re2 = r'[\d]{1,2}-[\d]{1,2}-[\d]{4}'
re3 = r'[\d]{1,2}.[\d]{1,2}.[\d]{4}'
re4 = r'[\d]{1,2} [ADFJMNOS]\w* [\d]{4}'
re5 = r'Date:[\d]{1,2}-[ADFJMNOS]\w*-[\d]{4}'
re6 = r'[\d]{1,2}/[ADFJMNOS]\w*/[\d]{4}'
re7 = r'Date:[\d]{1,2}/[ADFJMNOS]\w*/[\d]{4}'


generic_re = re.compile("(%s|%s|%s|%s|%s|%s|%s)" % (re1,re2,re3,re4,re5,re6,re7)).findall(data_cleaned)
	
#print(generic_re)

splitlines = text.splitlines( )
inv = re.compile(r'(receipt|receipt number|receipt no|invoice number|invoice no|inv|invoice|bill no|bill|bill id|invno|billid|billno|invoicenumber|invoiceno)\s*([:.-]+)?\s*[a-zA-Z0-9/\.]+[\d]',re.IGNORECASE)

for j in range(len(splitlines)):
	# splitlines[j]= re.sub("\s", "", splitlines[j])
	for i in inv.finditer(splitlines[j]):
		print ("......................",i.group(0))
		# print (splitlines)



for j in range(len(splitlines)):
	# print (splitlines[j])
	x = re.search(r"(pvt ltd|pvt|ltd|pvt. ltd|limited|llp)",splitlines[j],re.IGNORECASE)
	url = re.search(r'(www|WWW).[a-zA-Z0-9\.]*\b', splitlines[j])
	if (x):
		name = x.string
		print("...........................................................",name)
	else:
		x=""
	if(url):
		#print (splitlines[j])
		print("..........url...............",url.string)

	



