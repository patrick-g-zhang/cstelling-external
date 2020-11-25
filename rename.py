import json
import ipdb
import glob
import os

name_json_path = "/home/gyzhang/fastspeech2-master/checkpoints/fs2_ref_utt_story/generated_200000_/name_matchedList.json"
name_dicts = json.load(open(name_json_path))

wav_dir = "/home/gyzhang/speech_database/CSTelling/20200913_120655_clean_ph_lab/wavs"
generated_path = "/home/gyzhang/fastspeech2-master/checkpoints/fs2_ref_utt_story/generated_200000_/wavs/"

### process generate wavs #######
for wav_path in glob.glob(f'{wav_dir}/*.wav'):
    item_name = os.path.basename(wav_path)
    ipdb.set_trace()


for wav_path in glob.glob(f'{wav_dir}/*.wav'):
    item_name = os.path.basename(wav_path)
