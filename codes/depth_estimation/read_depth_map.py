import numpy as np
from tqdm import tqdm
import os
import cv2
save_dir = '/apdcephfs/share_916081/jarviswang/wt/code/SR/depth_estimation/monodepth2/results/Hamlyn/test'
npfile = '/apdcephfs/share_916081/jarviswang/wt/code/SR/depth_estimation/monodepth2/tmp/mono_model/models/weights_19/disps_endovis_split.npy'
pred_disp = np.load(npfile)

STEREO_SCALE_FACTOR = 5.4

num, h, w = pred_disp.shape
print(pred_disp.shape)
for i in tqdm(range(num)):
    depth = STEREO_SCALE_FACTOR / pred_disp[i,:,:]
    depth = np.clip(depth, 0, 80)
    depth = np.uint16(depth * 256)
    save_path = os.path.join(save_dir, "{:010d}.png".format(i))
    cv2.imwrite(save_path, depth)