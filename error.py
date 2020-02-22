from mcd import dtw
import mcd.metrics as mt


def mel_cep_dtw_dist(target, converted):
    """
    Compute the distance between two unaligned speech waveforms
    :param target: reference speech numpy array
    :param converted: synthesized speech numpy array
    :return: mel cep distance in dB
    """
    total_cost = 0
    total_frames = 0
    for (tar, conv) in zip(target, converted):
        tar, conv = tar.astype('float64'), conv.astype('float64')
        cost, _ = dtw.dtw(tar, conv, mt.logSpecDbDist)
        frames = len(tar)
        total_cost += cost
        total_frames += frames

    return total_cost / total_frames
