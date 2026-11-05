# PhysNet for rPPG Signal Reconstruction

This repository describes a video-based rPPG pipeline built around PhysNet, a spatiotemporal neural network for learning subtle pulse-related changes in facial appearance. The implementation uses normalized frame differences as video inputs, predicts a differential pulse waveform, combines predictions from overlapping temporal windows, and integrates the resulting signal to obtain a continuous rPPG waveform.

The reconstructed waveform is evaluated against a reference photoplethysmography (PPG) signal at two levels: waveform agreement and recording-level heart-rate estimation.


## Overview

Remote photoplethysmography estimates cardiovascular activity from video by analyzing small variations in light reflected from the skin. These variations contain a pulse-related component, but they are also affected by head motion, illumination changes, imaging conditions, and other nonphysiological factors.

Rather than estimating heart rate directly from RGB frames, this pipeline learns a time-dependent pulse representation. PhysNet processes short face-video sequences to predict the temporal changes of a reference PPG waveform. The differential predictions are then assembled into a recording-length signal, integrated, filtered, and analyzed in the frequency domain.

The main processing stages are:

1. Detect and crop the face in each RGB frame.
2. Repair temporary detection failures and reject recordings with excessive missing detections.
3. Form a normalized temporal-difference video representation.
4. Split the sequence into overlapping temporal windows.
5. Predict a differential pulse waveform using PhysNet.
6. Average overlapping predictions and cumulatively integrate them.
7. Apply identical postprocessing to the prediction and reference PPG.
8. Evaluate waveform correlation and estimate heart rate using spectral analysis.

## Methodology

### 1. Face localization and video preprocessing

Facial landmarks are detected separately for each video frame. The face region is cropped with an additional margin and resized to **72 × 72 pixels**, retaining its three RGB channels.

Temporary failures in face detection are handled by interpolating the missing crops. Recordings in which **more than 20%** of face detections are missing are excluded rather than reconstructed from a large number of unreliable frames.

### 2. Normalized temporal-difference representation

To emphasize frame-to-frame changes rather than static facial appearance, adjacent frames are transformed using an element-wise normalized difference:

$$
D_t = \frac{I_{t+1}-I_t}{I_{t+1}+I_t+\epsilon},
\qquad \epsilon = 10^{-7},
$$

where $I_t \in \mathbb{R}^{72\times72\times3}$ denotes the RGB crop at time $t$.


The resulting difference representation is scaled by its standard deviation over the video:

$$
\widetilde{D}_t = \frac{D_t}{\mathrm{std}(D)},
$$

where $\mathrm{std}(D)$ denotes the standard deviation used by the preprocessing implementation. Normalization reduces sensitivity to recording-dependent variation in the magnitude of pixel changes.

For supervised training, the reference PPG waveform $s_t$ is converted to a normalized first-order temporal difference:

$$
\Delta s_t = s_{t+1} - s_t,
\qquad
\widetilde{\Delta s}_t = \frac{\Delta s_t}{\mathrm{std}(\Delta s)}.
$$

Thus, the training target is a **differential pulse signal**, rather than the original PPG amplitude sequence. Frame differences are naturally indexed by adjacent-frame pairs; implementations must preserve this time indexing when aligning predictions and PPG targets.

### 3. Temporal windowing

The preprocessed video is divided into overlapping windows:

| Parameter | Value |
|---|---:|
| Nominal frame rate | 30 frames/s |
| Window length | 300 frames |
| Window stride | 150 frames |
| Nominal window duration | 10 s |
| Overlap between consecutive windows | 5 s (50%) |

Overlapping windows provide temporal context for the network and allow predictions from neighboring windows to be combined at inference time. Exact treatment of the first/last difference sample and incomplete terminal windows should remain consistent between training and inference.

### 4. PhysNet and learning objective

PhysNet was introduced by Yu, Li, and Zhao (BMVC 2019) as a spatiotemporal neural-network approach to recovering rPPG information from facial video. The present pipeline uses a PhysNet-based model to map RGB temporal-difference sequences to a one-dimensional differential pulse waveform.

In conceptual form,

$$
\hat{\mathbf d} = f_{\theta}(\widetilde{\mathbf D}),
$$

where $f_\theta$ is the learned network, $\widetilde{\mathbf D}$ is the temporal-difference video input, and $\hat{\mathbf d}$ is the predicted differential waveform.

The model is trained with a **Pearson-correlation-based objective** that permits a temporal alignment shift of up to **±0.33 seconds** between predictions and reference targets. Conceptually, a shift-tolerant correlation objective can be written as

$$
\mathcal{L}_{\mathrm{corr}}
= 1 - \max_{|\tau| \leq K}
\rho\left(
\hat{\mathbf{d}},
\mathrm{shift}(\widetilde{\Delta\mathbf{s}}, \tau)
\right),
$$

where $\rho$ is the Pearson correlation calculated on temporally overlapping samples and $K$ corresponds to approximately 0.33 seconds at the working sampling rate. This expresses the alignment-tolerant learning principle; the precise loss reduction and boundary handling should be verified against the training implementation.

This objective emphasizes similarity in the temporal shape of the differential waveform without requiring the predicted signal to match the reference amplitude scale.

### 5. Reconstruction of a continuous waveform

During inference, each window produces a differential pulse prediction. Because neighboring windows overlap, more than one prediction can correspond to the same time point.

These predictions are **uniformly averaged** over their shared time positions:

$$
\hat d_t =
\frac{1}{|\mathcal W_t|}
\sum_{w\in\mathcal W_t}\hat d_t^{(w)},
$$

where $\mathcal W_t$ is the set of windows containing time point $t$, and $\hat d_t^{(w)}$ is the prediction from window $w$ aligned to the recording timeline.

The full-length differential signal is subsequently integrated by cumulative summation:

$$
\hat s_0=0,
\qquad
\hat s_t=\sum_{k=0}^{t-1}\hat d_k.
$$

Cumulative integration recovers the waveform trajectory up to an arbitrary offset. It can also accumulate low-frequency errors, which motivates the detrending and filtering applied in the next stage.

### 6. Signal postprocessing

The reconstructed waveform and reference PPG undergo the same sequence of signal-processing steps:

1. Detrending to suppress slowly varying baseline components.
2. Second-order Butterworth band-pass filtering**, with a passband of 0.5–10 Hz.
3. Standardization to place the signals on a comparable scale.

The 0.5–10 Hz passband used for waveform processing is intentionally broader than the frequency range used to estimate heart rate. The wider band retains additional pulse-related frequency content, whereas heart-rate estimation searches specifically for the dominant fundamental-frequency component.

Predicted and reference waveforms must be sampled on a compatible time grid before calculating waveform-level metrics.

### 7. Heart-rate estimation

For each recording, mean heart rate is estimated using the dominant spectral peak in the range **0.7–3.0 Hz** (42–180 BPM):

$$
\hat f_{\mathrm{HR}}
=\underset{f\in[0.7,3.0]}{\arg\max}
\;\left|\mathcal F\{\hat s\}(f)\right|^2,
$$

$$
\widehat{\mathrm{HR}}=60\hat f_{\mathrm{HR}}\quad\text{BPM}.
$$

Here, $\mathcal F$ denotes the discrete Fourier transform of the processed waveform. Reference heart rate is estimated from the corresponding reference PPG recording using the same heart-rate search interval.

This procedure provides a single mean HR estimate per recording. It does not, by itself, evaluate instantaneous HR tracking, beat-to-beat intervals, or heart-rate variability.

## Training and evaluation data

The model-development sequence consists of:

| Stage | Dataset | Purpose |
|---|---|---|
| Initial training | [UBFC-rPPG](https://doi.org/10.1016/j.patrec.2017.10.017) | Learning pulse-related video features from synchronized facial video and reference signals |
| Fine-tuning | [MCD-rPPG](https://huggingface.co/datasets/akramic/mcd_rppg) | Further adaptation using additional recording conditions, including rest and post-exercise data |
| Independent evaluation | Private dataset | Evaluation on recordings not used for model training |

## Evaluation protocol

Evaluation is performed at the **recording level**, using both the reconstructed waveform and its associated reference PPG.

**Waveform agreement.** Pearson correlation is computed after applying the same signal-processing pipeline to the reconstructed and reference waveforms. Temporal alignment is optimized within **±0.33 s**. The reported mean correlation is the average of the recording-level correlations.

**Heart-rate error.** Let $\widehat{\mathrm{HR}}_i$ and $\mathrm{HR}_i$ denote estimated and reference mean heart rates for recording $i$. Define

$$
e_i=\widehat{\mathrm{HR}}_i-\mathrm{HR}_i.
$$

Mean absolute error and root mean squared error are calculated across recordings:

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|e_i|,
\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}.
$$

Each recording contributes one heart-rate error to these aggregate statistics. Therefore, the metrics describe **recording-level mean HR estimation**, not errors calculated independently for every video frame or temporal window.

## Results

The fine-tuned PhysNet-based model was evaluated on 8 independent recordings from a private dataset not used during training.

| Metric | Result |
|---|---:|
| Mean Pearson correlation (aligned waveforms) | 0.9015 |
| MAE of recording-level mean HR | 0.463 BPM |
| RMSE of recording-level mean HR | 0.724 BPM |
| Number of evaluated recordings | 8 |

## References and attribution

**Yu, Z., Li, X., & Zhao, G. (2019).** *Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks.* British Machine Vision Conference (BMVC). [Paper](https://arxiv.org/abs/1905.02419) · [Original PhysNet implementation](https://github.com/ZitongYu/PhysNet)

**Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019).** *Unsupervised skin tissue segmentation for remote photoplethysmography.* Pattern Recognition Letters, 124, 82–90. [DOI](https://doi.org/10.1016/j.patrec.2017.10.017)

**MCD-rPPG (2025).** *Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation.* Proceedings of ACM Multimedia 2025. [DOI](https://doi.org/10.1145/3746027.3758255) · [Dataset](https://huggingface.co/datasets/akramic/mcd_rppg)

