import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import json
import cv2
import os

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    # def _transform(self, n_px):
    #     return Compose([
    #         Resize((n_px,n_px), interpolation=Image.BICUBIC),
    #         # CenterCrop(n_px),
    #         lambda image: image.convert("RGB"),
    #         ToTensor(),
    #         # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #     ])

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    def get_video_data(self, total_video_data, frame_path, video_id, max_frames, start_time=None, end_time=None):
        
        images = []
        if video_id in total_video_data:
            frames = total_video_data[video_id]['frames']  # 프레임 파일명 리스트 (디렉토리는 없음)

            if len(frames) > 0:
                # `max_frames` 개수만큼 균등하게 선택
                selected_indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
                selected_frames = [frames[i] for i in selected_indices]

                for frame_name in selected_frames:
                    full_frame_path = frame_path + frame_name  # frame_path 붙이기
                    if os.path.exists(full_frame_path):  # 파일 존재하는지 확인
                        frame = Image.open(full_frame_path).convert("RGB")
                        images.append(self.transform(frame))  # 이미지 처리

        # 이미지 텐서 반환
        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)  # 이미지가 없을 경우

        return {'video': video_data}

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2