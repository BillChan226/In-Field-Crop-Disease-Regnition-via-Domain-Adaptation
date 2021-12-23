'''
debug 测试用的
'''
# %%
import skimage.io as io
import numpy as np
# %%
img = io.imread('./visulizaiton/ori.png', 1)
# %%
np.sum(img != 0)
# %%
img = io.imread('./visulizaiton/img.png', 1)
# %%
np.sum(img != 0)
# %%
