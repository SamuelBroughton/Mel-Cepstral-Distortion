import numpy as np
import librosa
import pyworld


def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=36):
    # Get Mel-cepstral coefficients (MCEPs)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp


def world_decode_spectral_envelop(coded_sp, fs):
    # Decode Mel-cepstral to sp
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp


def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav


def world_encode_wav(wav_file, fs, frame_period=5.0, coded_dim=36):
    wav = load_wav(wav_file, sr=fs)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)

    return f0, timeaxis, sp, ap, coded_sp


def log_f0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def coded_sp_statistics(coded_sps):
    # sp shape (T, D)
    coded_sps_concatenated = np.concatenate(coded_sps, axis=0)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=0, keepdims=False)
    coded_sps_std = np.std(coded_sps_concatenated, axis=0, keepdims=False)

    return coded_sps_mean, coded_sps_std