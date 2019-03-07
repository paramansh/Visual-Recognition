from PIL import Image
import os

size = 600, 400
files = os.listdir('test/')
for f in files:
    im = Image.open("test/" + f)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save("resized/" + f.split('.')[0] + ".jpg", "JPEG")
