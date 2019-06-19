from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cod import cod

def bbox(params):
    cx,cy,w,h = params
    x0,x1 = cx-w//2, cx+w//2
    y0,y1 = cy-h//2, cy+h//2
    plt.plot([x0,x0],[y0,y1],color='r',lw=0.5)
    plt.plot([x1,x1],[y0,y1],color='r',lw=0.5)
    plt.plot([x0,x1],[y0,y0],color='r',lw=0.5)
    plt.plot([x0,x1],[y1,y1],color='r',lw=0.5)

im = np.array(Image.open('test.png'))
objects = cod(im)

plt.imshow(im)
for i in objects:
    bbox(i)
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig('output.png', bbox_inches="tight", pad_inches=0)
plt.show()
