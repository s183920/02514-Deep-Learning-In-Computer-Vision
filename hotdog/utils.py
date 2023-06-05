import numpy as np



def data_to_img_array(data):
    return (data.numpy().transpose(0,2,3,1)*255).astype(np.uint8)