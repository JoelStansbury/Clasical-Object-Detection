from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cod import cod
from time import time
from skimage.color import rgb2hsv

def bbox(params):
    cx,cy,w,h = params
    x0,x1 = cx-w//2, cx+w//2
    y0,y1 = cy-h//2, cy+h//2
    plt.plot([x0,x0,x1,x1,x0],
             [y0,y1,y1,y0,y0],
             # color='r',
             lw=1)

im1 = np.array(Image.open('test.png'))
im = rgb2hsv(im1)
t0 = time()

# the black and white dial performs better with rgb
# while the landscape performs better with hsv
objects = cod(im1, Q = 2, eps=1,verbose=0)


print(time()-t0)
plt.imshow(im1)
for i in objects:
    bbox(i)
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig('output.png', bbox_inches="tight", pad_inches=0)
plt.show()
