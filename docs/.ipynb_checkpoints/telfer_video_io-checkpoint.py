import cv2
import numpy as np
from pathlib import Path

def get_fname(path):
    return Path(path).parts[-1].split('.')[0]

class VideoReader:
    def __init__(self, video):
        video = str(video)
        self.video = video
        self.fname = get_fname(video)
        self.cap = cv2.VideoCapture(video)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0
        
    def __getitem__(self, idx):
        if idx >= self.frames or idx < 0:
            raise IndexError("Out of bounds")
            
        if self.pos != idx:
            assert self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        
        ret, frame = self.cap.read()
        self.pos += 1
        frame = np.flip(frame, 2)
        return frame
    
    def __len__(self):
        return self.frames
    
    def __del__(self):
        self.cap.release()
    
class VideoWriter:
    def __init__(self, video, fps, width, height):
        self.video = str(video)
        self.fname = Path(self.video).parts[-1].split('.')[0]
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = None
        
    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.writer = cv2.VideoWriter(self.video, fourcc, self.fps, (self.width, self.height))
        return self
    
    def write(self, frame):
        frame = np.flip(frame)
        self.writer.write(frame)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()