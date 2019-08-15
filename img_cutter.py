import os
import time
from PIL import Image

j=0

path = '/home/val/google_recaptcha_set/3x3'

for root, dirs, files in os.walk(path):
   for name in files:

      img = Image.open(os.path.join(root, name))
   	new_image = img.crop((538, 235, 930, 627))
   	for i in range(9):
   		fin_img = new_image.crop(mini_pic_sizes[i])
   		fin_img.save('/home/val/google_recaptcha_set/recaptcha_set/3x3/train/pic_{}.jpg'.format(j))
   		#fin_img.show()
   		j += 1
   		#time.sleep(15)