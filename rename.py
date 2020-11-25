import json
import ipdb
import glob
import os
import re
import librosa
import numpy as np
import shutil
from utils.pwg_decode_from_mel import generate_wavegan, load_pwg_model

vocoder = load_pwg_model(
    config_path="/home/gyzhang/fastspeech2-master/wavegan_pretrained/config.yaml",
    checkpoint_path="/home/gyzhang/fastspeech2-master/wavegan_pretrained/checkpoint-1000000steps.pkl",
    stats_path="/home/gyzhang/fastspeech2-master/wavegan_pretrained/stats.h5",
)


def mel2wav(mel):
    wav_out = generate_wavegan(
        mel, *vocoder)
    wav_out = wav_out.cpu().numpy()
    return wav_out.T


def process_utterance(wav,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-10,
                      sample_rate=22050,
                      loud_norm=False,
                      min_level_db=-100,
                      return_linear=False,
                      trim_long_sil=False, vocoder='pwg'):

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin is -1 else fmin
    fmax = sample_rate / 2 if fmax is -1 else fmax
    mel_basis = librosa.filters.mel(
        sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    if vocoder == 'pwg':
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)

    return mel


def copy_wavs(tgt_dir, src_dir, index):
    os.makedirs(tgt_dir, exist_ok=True)
    for wav_path in glob.glob(f'{src_dir}/*.wav'):
        item_name = os.path.basename(wav_path)
        res_list = re.findall(r'\[(.*?)\]', item_name)
        assert len(res_list) == 2
        name_clip_name = res_list[-1]
        seg_name = name_clip_name[:-4]
        clips = re.split(r"\_", name_clip_name)
        name_in_seg_index = clips[-2]
        name_index = int(clips[-1])
        if name_index == index:
            shutil.copy(
                wav_path, f'{tgt_dir}/{seg_name}_{name_in_seg_index}.wav')


name_json_path = "/home/gyzhang/fastspeech2-master/checkpoints/fs2_ref_utt_story/generated_200000_/name_matchedList.json"
name_dicts = json.load(open(name_json_path))

wav_dir = "/home/gyzhang/speech_database/CSTelling/20200913_120655_clean_ph_lab/wavs"
generated_dir = "/home/gyzhang/fastspeech2-master/checkpoints/fs2_ref_utt_story/generated_200000_/wavs"
data_dir = "/home/gyzhang/fastspeech2-master/checkpoints/fs2_ref_utt_story/generated_200000_"

### process generate wavs #######
replace_name_amount = 2
name0_dir = f'{data_dir}/name0'
name1_dir = f'{data_dir}/name1'

# copy files
copy_wav_file = False
if copy_wav_file:
    copy_wavs(name0_dir, generated_dir, 0)
    copy_wavs(name1_dir, generated_dir, 1)


generated_names_dict = {}
for wav_path in glob.glob(f'{name0_dir}/*.wav'):
    item_name = os.path.basename(wav_path)[:-4]
    res_list = re.split("\_", item_name)
    name_in_seg_index = res_list[-1]
    seg_name = '_'.join(res_list[:-2])

    if generated_names_dict.get(seg_name) is None:
        generated_names_dict[seg_name] = dict()
    generated_names_dict[seg_name][name_in_seg_index] = wav_path


for wav_path in glob.glob(f'{wav_dir}/*.wav'):
    # check name exisits

    item_seg_name = os.path.basename(wav_path)[:-4]

    generated_name_dict = generated_names_dict.get(item_seg_name)

    if generated_name_dict is None:
        continue

    wav_raw, sr = librosa.core.load(wav_path, sr=22050)

    ori_names_list = [
        name_dict for name_dict in name_dicts if name_dict['seg_k'] == item_seg_name]
    # sort by start samples
    ori_names_list.sort(key=lambda x: x['start_sample'])

    # cut into different clips
    windices = []
    name_clips_indices = []
    for num, ori_name_dict in enumerate(ori_names_list):
        start_sample = ori_name_dict['start_sample']
        end_sample = ori_name_dict['end_sample']
        windices.append(start_sample)
        windices.append(end_sample)
        name_clips_indices.append(2 * num + 1)
    ipdb.set_trace()

    name_clips = np.split(wav_raw, windices)
    name_clips_mel = [process_utterance(name_clip) for name_clip in name_clips]

    ipdb.set_trace()
    wav_out = mel2wav(name_clips_mel[0])
    for name_index in range(len(ori_names_list)):
        wav_path = generated_name_dict[str(name_index)]
