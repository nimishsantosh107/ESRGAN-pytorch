# NOT WORKING (image transpose / format issue)

import cv2
import sys
import tqdm
import argparse
import torch
from PIL import Image
from model.ESRGAN import ESRGAN
import torchvision.transforms.functional as TF
import argparse


# Parse arguements
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gan_pth_path', 
    default='parameters/gan.pth'
)
parser.add_argument(
    '-i', 
    '--input',
    dest="input",
    default="C:\\Users\\nimish\\Programs\\Hippo\\Enhance\\INPUT\\large\\TEST.mp4", 
    type=str,
    help='Input Video Path'
)
parser.add_argument(
    '-o', 
    '--output',
    dest="output",
    default="C:\\Users\\nimish\\Programs\\Hippo\\Enhance\\esrgan_pytorch\\TEMP", 
    type=str,
    help='Output Video Path'
)
parser.add_argument(
    '-s', 
    '--resize-shape',
    dest="resize_shape",
    nargs='+',
    default=[270, 480], 
    type=int,
    help='Dimensions of preprocessed input video Eg: 270 480'
)
args = parser.parse_args()

dict_ESRGAN = torch.load(args.gan_pth_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Handle Video
video = cv2.VideoCapture(args.input)
output_pre = None
output_hr = None

if (video.isOpened() is False):
    print("Error opening video stream or file")
    sys.exit()

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

it=0

with torch.no_grad():
    esrgan = ESRGAN(3, 3, scale_factor=4)
    esrgan.load_state_dict(dict_ESRGAN)
    esrgan = esrgan.to(device).eval()

    pbar = tqdm.tqdm(total=total_frames)
    while(video.isOpened()):

        status, frame = video.read()
        if status is True:
            
            
            pre_frame = cv2.resize(frame, (32,32), interpolation = cv2.INTER_AREA)
            pre_frame_np = pre_frame
            pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2RGB)
            pre_frame = Image.fromarray(pre_frame)
            pre_frame_tensor = TF.to_tensor(pre_frame).to(device).unsqueeze(dim=0)
            hr_frame_tensor = esrgan(pre_frame_tensor)    

            hr_frame_np = hr_frame_tensor[0].cpu().numpy().transpose(2, 1, 0)        
            hr_frame_np = cv2.cvtColor(hr_frame_np, cv2.COLOR_RGB2BGR)
            
            # if (output_pre is None and output_hr is None):
            #     output_pre = cv2.VideoWriter(args.output+'_PRE.avi', cv2.VideoWriter_fourcc(*'MPEG'), 
            #         30, (int(pre_frame_np.shape[1]), int(pre_frame_np.shape[0])))
            #     output_hr = cv2.VideoWriter(args.output+'_HR.avi', cv2.VideoWriter_fourcc(*'MPEG'),
            #         30, (int(hr_frame_np.shape[1]), int(hr_frame_np.shape[0])))
            # output_pre.write(pre_frame_np)
            # output_hr.write(hr_frame_np)

            pbar.update(1)

            cv2.imwrite('C:\\Users\\nimish\\Programs\Hippo\\Enhance\\esrgan_pytorch\\DIR_LR\\%03d.png'%it, pre_frame_np)
            it += 1
            # cv2.imshow('Frame',frame)    
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        else: 
            break

pbar.close()
video.release()
output_pre.release()
output_hr.release()
cv2.destroyAllWindows()
