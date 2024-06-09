"""
@Author: ryan
"""
import glob
import numpy as np
import soundfile as sf
import os
import argparse
import configparser as CP
from tqdm import tqdm
from audiolib import audioread, audiowrite, snr_mixer

def setup_directories(cfg):
    base_dir = os.path.dirname(__file__)
    clean_dir = cfg["speech_dir"] if cfg["speech_dir"] != 'None' else os.path.join(base_dir, 'clean_train')
    noise_dir = cfg["noise_dir"] if cfg["noise_dir"] != 'None' else os.path.join(base_dir, 'noise_train')
    noisyspeech_dir = os.path.join(base_dir, 'NoisySpeech_training')
    clean_proc_dir = os.path.join(base_dir, 'CleanSpeech_training')
    noise_proc_dir = os.path.join(base_dir, 'Noise_training')

    for dir_path in [clean_dir, noise_dir, noisyspeech_dir, clean_proc_dir, noise_proc_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    return clean_dir, noise_dir, noisyspeech_dir, clean_proc_dir, noise_proc_dir

def read_configuration(cfg_path, cfg_section):
    assert os.path.exists(cfg_path), f"No configuration file found at {cfg_path}"
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfg_path)
    return cfg._sections[cfg_section]

def prepare_audio_files(clean_dir, noise_dir, cfg):
    audioformat = cfg["audioformat"]
    clean_files = glob.glob(os.path.join(clean_dir, audioformat))
    noise_files = glob.glob(os.path.join(noise_dir, audioformat))
    
    if cfg["noise_types_excluded"] != 'None':
        exclude_list = cfg["noise_types_excluded"].split(',')
        noise_files = [fn for fn in noise_files if not any(fn.startswith(excl) for excl in exclude_list)]
    
    if not clean_files:
        raise FileNotFoundError(f"No clean audio files found in {clean_dir} matching format {audioformat}")
    
    if not noise_files:
        raise FileNotFoundError(f"No noise audio files found in {noise_dir} matching format {audioformat}")

    return clean_files, noise_files

def process_audio_files(clean_files, noise_files, cfg, SNR, noisyspeech_dir, clean_proc_dir, noise_proc_dir, total_hours):
    total_samples = int(total_hours * 3600 * float(cfg["sampling_rate"]))
    audio_length = int(float(cfg["audio_length"]) * float(cfg["sampling_rate"]))
    silence_length = float(cfg["silence_length"])
    filecounter = 0
    num_samples = 0

    # Create a progress bar
    with tqdm(total=total_samples, desc="Generating noisy speech data") as pbar:
        while num_samples < total_samples:
            clean, fs = fetch_random_file(clean_files, audio_length, silence_length)
            noise, _ = fetch_random_file(noise_files, len(clean), silence_length)

            filecounter += 1
            for snr in SNR:
                clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=snr)
                save_audio_files(clean_snr, noise_snr, noisy_snr, fs, filecounter, snr, noisyspeech_dir, clean_proc_dir, noise_proc_dir)
                num_samples += len(noisy_snr)
                pbar.update(len(noisy_snr))

def fetch_random_file(file_list, length_in_samples, silence_length):
    if not file_list:
        raise ValueError("file_list is empty. Ensure that the directory contains the necessary files.")

    idx = np.random.randint(0, len(file_list))
    audio, fs = audioread(file_list[idx])

    while len(audio) <= length_in_samples:
        idx = (idx + 1) % len(file_list)
        new_audio, _ = audioread(file_list[idx])
        audio = np.append(audio, np.zeros(int(fs * silence_length)))
        audio = np.append(audio, new_audio)

    return audio[:length_in_samples], fs

def save_audio_files(clean, noise, noisy, fs, filecounter, snr, noisyspeech_dir, clean_proc_dir, noise_proc_dir):
    noisyfilename = f'noisy{filecounter}_SNRdb_{snr}_clnsp{filecounter}.wav'
    cleanfilename = f'clnsp{filecounter}.wav'
    noisefilename = f'noisy{filecounter}_SNRdb_{snr}.wav'
    noisypath = os.path.join(noisyspeech_dir, noisyfilename)
    cleanpath = os.path.join(clean_proc_dir, cleanfilename)
    noisepath = os.path.join(noise_proc_dir, noisefilename)

    audiowrite(noisy, fs, noisypath, norm=False)
    audiowrite(clean, fs, cleanpath, norm=False)
    audiowrite(noise, fs, noisepath, norm=False)

def main(cfg, total_hours):
    clean_dir, noise_dir, noisyspeech_dir, clean_proc_dir, noise_proc_dir = setup_directories(cfg)
    
    # Set noise level range from 10 to 40 dB
    SNR = np.linspace(10, 40, int(cfg["total_snrlevels"]))
    clean_files, noise_files = prepare_audio_files(clean_dir, noise_dir, cfg)

    # Process each clip in clean_train to be made noisy
    process_audio_files(clean_files, noise_files, cfg, SNR, noisyspeech_dir, clean_proc_dir, noise_proc_dir, total_hours)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="noisyspeech_synthesizer.cfg", help="Read noisyspeech_synthesizer.cfg for all the details")
    parser.add_argument("--cfg_str", type=str, default="noisy_speech")
    parser.add_argument("--total_hours", type=float, required=True, help="Total hours of data to be created")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), args.cfg)
    config = read_configuration(cfg_path, args.cfg_str)

    # Override total_hours with command-line argument
    total_hours = args.total_hours

    main(config, total_hours)
