import os
import json
import pickle
        
        
def navi_format_data():
    train_list, val_list, test_list = "", "", ""
    path = "./data_navi/data/train-tracks.json"
    data = {}
    with open(path, "r") as f:
        data = json.load(f)

    aicity_dict = {}
    for idx, vid in enumerate(data):
        info = data[vid]
        new_info = []
        
        for sentence in info['nl']:
            words = sentence.strip(".").split()
            if words:
                new_info.append(words)
               

        aicity_dict[vid] = new_info

        # train_list += str(vid) + "\n"
        

        # if you want to make a validation set, uncomment the following lines
        # randomly split the train and val
        if idx % 20 == 0:
            val_list += str(vid) + "\n"
        else:
            train_list += str(vid) + "\n"





    txt_path = "./data_navi/data/test-queries.json"
    txt_data = {}
    with open(txt_path, "r") as f:
        txt_data = json.load(f)

    video_path = "./data_navi/data/test-tracks.json"
    video_data = {}
    with open(video_path, "r") as f:
        video_data = json.load(f)

    for tid, vid in zip(txt_data.keys(), video_data.keys()):  
        info = txt_data[tid]
        new_info = []
        
        for sentence in info['nl']:
            words = sentence.strip(".").split()
            if words:

                new_info.append(words)
        aicity_dict[vid] = new_info
        test_list += str(vid) + "\n" 
        

    with open('./data_navi/navi/input/raw-captions.pkl', 'wb') as handle:
        pickle.dump(aicity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    with open('./data_navi/navi/input/train_list.txt', 'w') as f:
        f.write(train_list)
    with open('./data_navi/navi/input/val_list.txt', 'w') as f:
        f.write(val_list)
    with open('./data_navi/navi/input/test_list.txt', 'w') as f:
        f.write(test_list)



navi_format_data()
