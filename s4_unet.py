
# %%
from keras.layers.convolutional import Conv2D
from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Concatenate, Activation
from keras.optimizers import Adam
from keras import initializers
from keras.models import Model
from keras import backend as K
from memory_profiler import profile


#%%
def iu_acc(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[1, 2, 3], keepdims=False)
    sum_ = K.sum(y_true + y_pred_pos, axis=[1, 2, 3], keepdims=False)
    jac = (intersection) / (sum_ - intersection + smooth)
    return K.mean(jac)

def dice_acc(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[1, 2, 3], keepdims=False)
    sum_ = K.sum(y_true + y_pred_pos, axis=[1, 2, 3], keepdims=False)
    jac = (2*intersection + smooth) / (sum_ + smooth)
    return K.mean(jac)

# loss function
def my_loss(y_true,y_pred):
    bc = K.binary_crossentropy(y_pred, y_true)
    false_bc = (1.0 - y_true) * bc
    true_bc = y_true * bc
    mloss = false_bc + 10.0 * true_bc
    return mloss

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# dice_coef may have better performance
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
#%%
def ConvActive(np_ker, size, inputs, 
                        initial = initializers.glorot_normal(seed=None)):
    
    return Conv2D(np_ker, size, activation='relu',padding = 'same', kernel_initializer = initial)(inputs)

def ConvBatchnormActive(np_ker, size, inputs, 
                        initial = initializers.glorot_normal(seed=None)):
    
    return Activation('relu')(BatchNormalization()(Conv2D(np_ker, size,padding = 'same', kernel_initializer = initial)(inputs)))

def UpConvActiveContact(np_ker, size, inputs, contact,
                  initial = initializers.glorot_normal(seed=None)):
    up = UpSampling2D(size = (2,2))(inputs)
    upconv = Conv2D(np_ker, size, activation = 'relu', padding = 'same', kernel_initializer = initial)(up) 
    return Concatenate(axis=3)([upconv, contact])


def get_unet(pretrained_weights = None):
    
    inputs = Input((512, 512, 3))

    conv1 = ConvBatchnormActive(16, 3, inputs)
    conv1 = ConvBatchnormActive(16, 3, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = ConvBatchnormActive(32, 3, pool1)
    conv2 = ConvBatchnormActive(32, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = ConvBatchnormActive(64, 3, pool2)
    conv3 = ConvBatchnormActive(64, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = ConvBatchnormActive(128, 3, pool3)
    conv4 = ConvBatchnormActive(128, 3, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = ConvBatchnormActive(256, 3, pool4)
    conv5 = ConvBatchnormActive(256, 3, conv5)
    
    up6 = UpConvActiveContact(128, 2, conv5, conv4)
    conv6 = ConvBatchnormActive(128, 3, up6)
    conv6 = ConvBatchnormActive(128, 3, conv6)
    
    up7 = UpConvActiveContact(64, 2, conv6, conv3)
    conv7 = ConvBatchnormActive(64, 3, up7)
    conv7 = ConvBatchnormActive(64, 3, conv7)
    
    up8 = UpConvActiveContact(32, 2, conv7, conv2)
    conv8 = ConvBatchnormActive(32, 3, up8)
    conv8 = ConvBatchnormActive(32, 3, conv8)
    
    up9 = UpConvActiveContact(16, 2, conv8, conv1)
    conv9 = ConvBatchnormActive(16, 3, up9)
    conv9 = ConvBatchnormActive(16, 3, conv9)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [iu_acc, dice_acc, 'accuracy'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


#%% training 
@profile
def main():
    model = get_unet()
    ans = model.summary()

if __name__ == '__main__':
    main()

# # %%
# ans = '''
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, 512, 512, 3)       0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 512, 512, 16)      448       
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 512, 512, 16)      64        
# _________________________________________________________________
# activation_1 (Activation)    (None, 512, 512, 16)      0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 512, 512, 16)      2320      
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 512, 512, 16)      64        
# _________________________________________________________________
# activation_2 (Activation)    (None, 512, 512, 16)      0         
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 256, 256, 16)      0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 256, 256, 32)      4640      
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 256, 256, 32)      128       
# _________________________________________________________________
# activation_3 (Activation)    (None, 256, 256, 32)      0         
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 256, 256, 32)      9248      
# _________________________________________________________________
# batch_normalization_4 (Batch (None, 256, 256, 32)      128       
# _________________________________________________________________
# activation_4 (Activation)    (None, 256, 256, 32)      0         
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 128, 128, 32)      0         
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 128, 128, 64)      18496     
# _________________________________________________________________
# batch_normalization_5 (Batch (None, 128, 128, 64)      256       
# _________________________________________________________________
# activation_5 (Activation)    (None, 128, 128, 64)      0         
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 128, 128, 64)      36928     
# _________________________________________________________________
# batch_normalization_6 (Batch (None, 128, 128, 64)      256       
# _________________________________________________________________
# activation_6 (Activation)    (None, 128, 128, 64)      0         
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 64, 64, 64)        0         
# _________________________________________________________________
# conv2d_7 (Conv2D)            (None, 64, 64, 128)       73856     
# _________________________________________________________________
# batch_normalization_7 (Batch (None, 64, 64, 128)       512       
# _________________________________________________________________
# activation_7 (Activation)    (None, 64, 64, 128)       0         
# _________________________________________________________________
# conv2d_8 (Conv2D)            (None, 64, 64, 128)       147584    
# _________________________________________________________________
# batch_normalization_8 (Batch (None, 64, 64, 128)       512       
# _________________________________________________________________
# activation_8 (Activation)    (None, 64, 64, 128)       0         
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 32, 32, 128)       0         
# _________________________________________________________________
# conv2d_9 (Conv2D)            (None, 32, 32, 256)       295168    
# _________________________________________________________________
# batch_normalization_9 (Batch (None, 32, 32, 256)       1024      
# _________________________________________________________________
# activation_9 (Activation)    (None, 32, 32, 256)       0         
# _________________________________________________________________
# conv2d_10 (Conv2D)           (None, 32, 32, 256)       590080    
# _________________________________________________________________
# batch_normalization_10 (Batc (None, 32, 32, 256)       1024      
# _________________________________________________________________
# activation_10 (Activation)   (None, 32, 32, 256)       0         
# _________________________________________________________________
# up_sampling2d_1 (UpSampling2 (None, 64, 64, 256)       0         
# _________________________________________________________________
# conv2d_11 (Conv2D)           (None, 64, 64, 128)       131200    
# _________________________________________________________________
# concatenate_1 (Concatenate)  (None, 64, 64, 256)       0         
# _________________________________________________________________
# conv2d_12 (Conv2D)           (None, 64, 64, 128)       295040    
# _________________________________________________________________
# batch_normalization_11 (Batc (None, 64, 64, 128)       512       
# _________________________________________________________________
# activation_11 (Activation)   (None, 64, 64, 128)       0         
# _________________________________________________________________
# conv2d_13 (Conv2D)           (None, 64, 64, 128)       147584    
# _________________________________________________________________
# batch_normalization_12 (Batc (None, 64, 64, 128)       512       
# _________________________________________________________________
# activation_12 (Activation)   (None, 64, 64, 128)       0         
# _________________________________________________________________
# up_sampling2d_2 (UpSampling2 (None, 128, 128, 128)     0         
# _________________________________________________________________
# conv2d_14 (Conv2D)           (None, 128, 128, 64)      32832     
# _________________________________________________________________
# concatenate_2 (Concatenate)  (None, 128, 128, 128)     0         
# _________________________________________________________________
# conv2d_15 (Conv2D)           (None, 128, 128, 64)      73792     
# _________________________________________________________________
# batch_normalization_13 (Batc (None, 128, 128, 64)      256       
# _________________________________________________________________
# activation_13 (Activation)   (None, 128, 128, 64)      0         
# _________________________________________________________________
# conv2d_16 (Conv2D)           (None, 128, 128, 64)      36928     
# _________________________________________________________________
# batch_normalization_14 (Batc (None, 128, 128, 64)      256       
# _________________________________________________________________
# activation_14 (Activation)   (None, 128, 128, 64)      0         
# _________________________________________________________________
# up_sampling2d_3 (UpSampling2 (None, 256, 256, 64)      0         
# _________________________________________________________________
# conv2d_17 (Conv2D)           (None, 256, 256, 32)      8224      
# _________________________________________________________________
# concatenate_3 (Concatenate)  (None, 256, 256, 64)      0         
# _________________________________________________________________
# conv2d_18 (Conv2D)           (None, 256, 256, 32)      18464     
# _________________________________________________________________
# batch_normalization_15 (Batc (None, 256, 256, 32)      128       
# _________________________________________________________________
# activation_15 (Activation)   (None, 256, 256, 32)      0         
# _________________________________________________________________
# conv2d_19 (Conv2D)           (None, 256, 256, 32)      9248      
# _________________________________________________________________
# batch_normalization_16 (Batc (None, 256, 256, 32)      128       
# _________________________________________________________________
# activation_16 (Activation)   (None, 256, 256, 32)      0         
# _________________________________________________________________
# up_sampling2d_4 (UpSampling2 (None, 512, 512, 32)      0         
# _________________________________________________________________
# conv2d_20 (Conv2D)           (None, 512, 512, 16)      2064      
# _________________________________________________________________
# concatenate_4 (Concatenate)  (None, 512, 512, 32)      0         
# _________________________________________________________________
# conv2d_21 (Conv2D)           (None, 512, 512, 16)      4624      
# _________________________________________________________________
# batch_normalization_17 (Batc (None, 512, 512, 16)      64        
# _________________________________________________________________
# activation_17 (Activation)   (None, 512, 512, 16)      0         
# _________________________________________________________________
# conv2d_22 (Conv2D)           (None, 512, 512, 16)      2320      
# _________________________________________________________________
# batch_normalization_18 (Batc (None, 512, 512, 16)      64        
# _________________________________________________________________
# activation_18 (Activation)   (None, 512, 512, 16)      0         
# _________________________________________________________________
# conv2d_23 (Conv2D)           (None, 512, 512, 1)       17        
# =================================================================
# Total params: 1,946,993.0
# Trainable params: 1,944,049.0
# Non-trainable params: 2,944.0
# _________________________________________________________________
# '''

# # %%
# bns = ans.split('\n')

# # %%
# cns = []
# for i in bns:
#     if i == '':
#         continue
#     if i[0] not in ['_', '=', 'L', 'T', 'N']:
#         cns.append(i)

# # %%
# dns = ''
# for s in cns:
#     pos1 = s.rfind('_')
#     name = s[0:pos1]
#     params = s.split()[-1]
#     pos3, pos2 = s.rfind(')'), s.rfind('(')
#     shape = s[pos2:pos3+1]
#     dns += name+'&'+shape+'&'+params+'.'

# # %%
