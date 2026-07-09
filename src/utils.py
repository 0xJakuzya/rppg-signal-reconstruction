import random
import numpy as np
import scipy.signal
import scipy.sparse
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
    identity = np.eye(length)
    ones = np.ones(length)
    diagonals = np.array([ones, -2 * ones, ones])
    second_difference = scipy.sparse.spdiags(diagonals, [0, 1, 2], length - 2, length).toarray()
    smooth = np.linalg.inv(identity + smoothing ** 2 * second_difference.T @ second_difference)
    return (identity - smooth) @ signal.astype(np.float64)

def bandpass_filter(signal, fps, low_hz=config.band_lo_hz, high_hz=config.band_hi_hz):
    nyquist = fps / 2.0
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.999)
    numerator, denominator = scipy.signal.cheby2(
        config.cheby_order, config.cheby_rs, [low, high], btype="bandpass"
    )
    filtered = scipy.signal.filtfilt(numerator, denominator, signal.astype(np.float64))
    return filtered.astype(np.float32)

def fft_hr(signal, fps, low_hz=config.hr_lo_hz, high_hz=config.hr_hi_hz):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()
    fft_length = 2 ** int(np.ceil(np.log2(len(signal))))
    frequencies = np.fft.rfftfreq(fft_length, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal, n=fft_length)) ** 2
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_frequencies = frequencies[in_band]
    band_power = power[in_band]
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
    best = -1.0
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
            best = max(best, float((a * b).sum() / denom))
    return best

def hr_metrics(predictions, targets, fps, max_shift=0):
    errors = []
    correlations = []
    for prediction, target in zip(predictions, targets):
        predicted_hr = fft_hr(prediction, fps)
        true_hr = fft_hr(target, fps)
        if np.isfinite(predicted_hr) and np.isfinite(true_hr):
            errors.append(predicted_hr - true_hr)
        if max_shift > 0:
            correlations.append(shift_corr(prediction, target, max_shift))
    errors = np.array(errors)
    mean_absolute_error = float(np.mean(np.abs(errors)))
    root_mean_squared_error = float(np.sqrt(np.mean(errors ** 2)))
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
