import json
import ipdb
import glob
import os
import re

name_json_path = "/home/gyzhang/fastspeech2-master/checkpoints/fs2_ref_utt_story/generated_200000_/name_matchedList.json"
name_dicts = json.load(open(name_json_path))

wav_dir = "/home/gyzhang/speech_database/CSTelling/20200913_120655_clean_ph_lab/wavs"
generated_path = "/home/gyzhang/fastspeech2-master/checkpoints/fs2_ref_utt_story/generated_200000_/wavs/"

### process generate wavs #######

generated_names_dict = {}
for wav_path in glob.glob(f'{generated_path}/*.wav'):
    item_name = os.path.basename(wav_path)
    res_list = re.findall(r'\[(.*?)\]', item_name)
    assert len(res_list) == 2
    name_clip_name = res_list[-1]
    seg_name = name_clip_name[:-4]
    clips = re.split(r"\_", name_clip_name)
    name_clip_in_seg_index = clips[-2]
    name_index = clips[-1]
    generated_names_dict[name_clip_name] = [
        seg_name, name_clip_in_seg_index, name_index]


for k, v in generated_names_dict.items():
    if '大小白鷺' in k:
        ipdb.set_trace()


for wav_path in glob.glob(f'{wav_dir}/*.wav'):
    item_seg_name = os.path.basename(wav_path)
    generated_name_list = [
        v for k, v in generated_names_dict.items() if item_seg_name in k]
    if len(generated_name_list) == 0:
        continue
    ipdb.set_trace()
