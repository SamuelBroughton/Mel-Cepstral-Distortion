import argparse
import os
from glob import glob
from utils import *
from tqdm import tqdm


def gen_spk_world_features(spk_path, out_dir, sample_rate=16000):
    """

    :param spk_path: path to speaker wav dir
    :param out_dir: path to output
    :param sample_rate:
    :return:
    """
    spk_paths = glob(os.path.join(spk_path, '*.wav'))
    spk_name = os.path.basename(spk_path)
    f0s = []
    coded_sps = []
    for wav_file in spk_paths:
        f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0s.append(f0)
        coded_sps.append(coded_sp)

    log_f0s_mean, log_f0s_std = log_f0_statistics(f0s)
    coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)

    np.savez(os.path.join(out_dir, spk_name + '_stats.npz'),
             log_f0s_mean=log_f0s_mean,
             log_f0s_std=log_f0s_std,
             coded_sps_mean=coded_sps_mean,
             coded_sps_std=coded_sps_std)

    for wav_file in tqdm(spk_paths):
        wav_name = os.path.basename(wav_file)
        _, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normalised_coded_sp = (coded_sp - coded_sps_mean) / coded_sps_std
        np.save(os.path.join(out_dir, wav_name.replace('.wav', '.npy')),
                normalised_coded_sp,
                allow_pickle=False)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data_dir_default = './data'
    output_dir_default = './data/processed'

    parser.add_argument('-d', '--data_dir', type=str, default=data_dir_default, help='Speakers directory location.')
    parser.add_argument('-s', '--speaker_dirs', type=str, nargs='+', required=True, help='Speakers to be processed.')
    parser.add_argument('-o', '--output_dir', type=str, default=output_dir_default, help='Processed speakers location.')
    #  TODO: add resampling capabilities
    # parser.add_argument('-r', '--resample', type=int, help='Resampling rate.')

    config = parser.parse_args()

    os.makedirs(config.output_dir, exist_ok=True)

    for spk in config.speaker_dirs:
        print('Processing: ', spk)
        spk_dir = os.path.join(config.data_dir, spk)
        gen_spk_world_features(spk_dir, config.output_dir)

    print('Complete, you can find processed speakers in ', config.output_dir)
