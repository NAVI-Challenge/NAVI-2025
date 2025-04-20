import json
import numpy as np
import zipfile
import io
import os

video_json = "./data_navi/data/test-tracks.json"
text_json = "./data_navi/data/test-queries.json"
sim_matrix_path = "path/to/your/sim/matrix/.npy"


sim_matrix = np.load(sim_matrix_path).squeeze()

with open(video_json, "r") as f:
    video_dict = json.load(f)

with open(text_json, "r") as f:
    text_dict = json.load(f)

video_uuids = list(video_dict.keys())
text_uuids = list(text_dict.keys())

text_to_videos = {}
for text_idx, text_uuid in enumerate(text_uuids):
    sim_scores = sim_matrix[text_idx]
    sorted_vid_indices = np.argsort(sim_scores)[::-1]
    text_to_videos[text_uuid] = [video_uuids[i] for i in sorted_vid_indices]

json_bytes = json.dumps(text_to_videos, indent=4).encode('utf-8')

zip_file_path = "./data_navi/navi/output/submissions/submission.zip"
os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.writestr("submission.json", json_bytes)

print(f"Saved Submission File to {zip_file_path}")
