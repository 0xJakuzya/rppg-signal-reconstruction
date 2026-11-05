import random
import numpy as np
import scipy.signal
import scipy.sparse
import scipy.sparse.linalg
import torch
from src import config

def normalize_signal(signal):
    signal = signal.astype(np.float32)
    signal = signal - signal.mean()
    standard_deviation = signal.std()
    if standard_deviation > config.eps:
        signal = signal / standard_deviation
    return signal.astype(np.float32)

def detrend(signal, smoothing=config.detrend_lambda):
    length = len(signal)
    if length < 3:
        return signal.astype(np.float64)
    identity = scipy.sparse.eye(length, format="csc")
    ones = np.ones(length - 2)
    second_difference = scipy.sparse.diags([ones, -2 * ones, ones], [0, 1, 2], shape=(length - 2, length), format="csc")
    smooth = scipy.sparse.linalg.spsolve(identity + smoothing ** 2 * second_difference.T @ second_difference, signal.astype(np.float64))
    return signal.astype(np.float64) - smooth

def bandpass_filter(signal, fps, low_hz=config.band_lo_hz, high_hz=config.band_hi_hz):
    if config.filter_type != "butterworth":
        print(f"unsupported filter_type: {config.filter_type}")
        return None
    if not 0 < low_hz < high_hz < fps / 2.0:
        print(f"invalid filter band {low_hz}-{high_hz} Hz for {fps} fps")
        return None
    sections = scipy.signal.butter(config.filter_order, [low_hz, high_hz], btype="bandpass", fs=fps, output="sos")
    filtered = scipy.signal.sosfiltfilt(sections, signal.astype(np.float64))
    return filtered.astype(np.float32)

def fft_hr(signal, fps, low_hz=config.hr_lo_hz, high_hz=config.hr_hi_hz):
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 2 or not np.isfinite(signal).all() or signal.std() <= config.eps:
        return float("nan")
    signal = signal - signal.mean()
    fft_length = 2 ** int(np.ceil(np.log2(len(signal))))
    frequencies = np.fft.rfftfreq(fft_length, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal, n=fft_length)) ** 2
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_frequencies = frequencies[in_band]
    band_power = power[in_band]
    if len(band_power) == 0:
        return float("nan")
    peak_index = band_power.argmax()
    peak_freq = band_frequencies[peak_index]
    half_freq = peak_freq / 2.0
    if half_freq >= low_hz:
        near_half = np.abs(band_frequencies - half_freq) <= 0.1
        if near_half.any() and band_power[near_half].max() >= 0.5 * band_power[peak_index]:
            peak_freq = half_freq
    heart_rate = peak_freq * 60.0
    return float(heart_rate)

def shift_corr(prediction, target, max_shift):
    length = len(prediction)
    best = float("nan")
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a, b = prediction[shift:], target[: length - shift]
        else:
            a, b = prediction[:shift], target[-shift:]
        if len(a) < 2:
            continue
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom > 0:
            correlation = float((a * b).sum() / denom)
            if not np.isfinite(best) or correlation > best:
                best = correlation
    return best

def hr_metrics(predictions, targets, fps, max_shift=0):
    errors = []
    correlations = []
    fps = np.broadcast_to(fps, (len(predictions),))
    max_shift = np.broadcast_to(max_shift, (len(predictions),))
    for prediction, target, sample_fps, shift in zip(predictions, targets, fps, max_shift):
        predicted_hr = fft_hr(prediction, sample_fps)
        true_hr = fft_hr(target, sample_fps)
        if np.isfinite(predicted_hr) and np.isfinite(true_hr):
            errors.append(predicted_hr - true_hr)
        correlation = shift_corr(prediction, target, int(shift))
        if np.isfinite(correlation):
            correlations.append(correlation)
    errors = np.array(errors)
    mean_absolute_error = float(np.mean(np.abs(errors))) if len(errors) else float("nan")
    root_mean_squared_error = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan")
    metrics = {
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
        "n": len(errors),
        "corr": float(np.mean(correlations)) if correlations else float("nan"),
    }
    return metrics

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
