from scipy.interpolate import CubicSpline
import numpy as np
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import utils
from tqdm import tqdm
import cv2
import random
import sys
import math
from model import StyledGenerator
from generate import get_mean_style

standard_normal_distribution = torch.distributions.normal.Normal(0, 1)

RESOLUTION = 256
STEP = int(math.log(RESOLUTION, 2)) - 2

DURATION_IN_SECONDS = 60
SAMPLE_COUNT = 30 # Number of distinct objects to generate and interpolate between
TRANSITION_FRAMES = DURATION_IN_SECONDS * 30 // SAMPLE_COUNT

LATENT_CODE_SIZE = 512

TILES = (3, 3)

generator = StyledGenerator(LATENT_CODE_SIZE).to(device)
generator.load_state_dict(torch.load('checkpoint/train_step-7.model')['g_running'])
generator.eval()

@torch.no_grad()
def get_spline(use_styles=True):
    codes = standard_normal_distribution.sample((SAMPLE_COUNT + 1, LATENT_CODE_SIZE))
    if use_styles:
        codes = generator.style(codes.to(device))
    
    codes[0, :] = codes[-1, :] # Make animation periodic
    return CubicSpline(np.arange(SAMPLE_COUNT + 1), codes.detach().cpu().numpy(), axis=0, bc_type='periodic')

def get_noise():
    noise = []

    for i in range(STEP + 1):
        size = 4 * 2 ** i
        noise.append(torch.randn(1, 1, size, size, device=device))

    return noise

splines = [get_spline() for i in range(TILES[0] * TILES[1])]
noises = [get_noise() for i in range(TILES[0] * TILES[1])]

@torch.no_grad()
def create_image_sequence():
    frame_index = 0
    progress_bar = tqdm(total=SAMPLE_COUNT * TRANSITION_FRAMES)

    
    for sample_index in range(SAMPLE_COUNT):
        for step in range(TRANSITION_FRAMES):
            images = []
            for spline, noise in zip(splines, noises):
                code = torch.tensor(spline(float(sample_index) + step / TRANSITION_FRAMES), dtype=torch.float32, device=device).reshape(1, -1)
                image = generator.generator([code], noise=noise, step=STEP, alpha=1)
                images.append(image)

            result = torch.zeros((3, TILES[0] * RESOLUTION, TILES[1] * RESOLUTION), device=device)
            p = 0
            for x in range(TILES[0]):
                for y in range(TILES[1]):
                    result[:, x * RESOLUTION:(x+1)*RESOLUTION, y * RESOLUTION:(y+1)*RESOLUTION] = images[p]
                    p += 1
                
            utils.save_image(result, 'images/frame-{:05d}.png'.format(frame_index), normalize=True, range=(-1, 1))
            
            frame_index += 1
            progress_bar.update()
    
    print("\n\nUse this command to create a video:\n")
    print('ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4')


mean_style = get_mean_style(generator, device)
create_image_sequence()