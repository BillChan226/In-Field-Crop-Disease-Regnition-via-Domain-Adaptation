'''
绘制超像素的图片
'''

#%%
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import mark_boundaries
import numpy as np
#%%
img = io.imread('../dataset/plant_seg/test/img/4.png')

# %%
segments = slic(img, n_segments = 500, sigma = 3)
# %%
img2 = mark_boundaries(img, segments, color=(1,1,0))
plt.figure()
plt.axis('off')
plt.imshow(img2)
plt.savefig('super.png', format='png', dpi=400, bbox_inches='tight')
# %%

# %%
