
# PhysNet: Transfer Learning from UBFC-rPPG to MCD-rPPG

This repository contains pretrained PhysNet weights for remote photoplethysmography (rPPG) signal reconstruction from facial RGB videos.

The project investigates transfer learning from UBFC-rPPG to MCD-rPPG, followed by evaluation on an independent private dataset containing recordings from 8 subjects.

The model predicts a differential pulse waveform, which is integrated and postprocessed to reconstruct the continuous blood volume pulse (BVP) signal.

## Datasets

| Dataset | Purpose | Data used |
|---|---|---|
| UBFC-rPPG | Pretraining | 42 subjects |
| MCD-rPPG | Fine-tuning | FullHDwebcam, 540 subjects |
| Private dataset | Test | 8 subjects |

## Training

Faces are detected using MediaPipe Face Landmarker. Each frame is cropped to the facial region and resized to 72 × 72 pixels, preserving the RGB color representation.

The model uses a DiffNormalized input representation:

$$
D_t = \frac{I_{t+1} - I_t}{I_{t+1} + I_t + 10^{-7}}
$$

where \\(I_t\\) and \\(I_{t+1}\\) represent consecutive RGB frames.

The frame differences are subsequently normalized by the standard deviation calculated over the entire video. The reference PPG target is represented by its normalized first-order temporal difference.

Preprocessed recordings are divided into:

- Window duration: 10 seconds
- Sampling frequency: 30 Hz
- Window stride: 5 seconds

For MCD-rPPG, only recordings from the FullHDwebcam camera with an original frame rate of at least 27 Hz are used.

## Evaluation protocol

Evaluation is performed at the video level, rather than independently on overlapping windows.

The evaluation pipeline consists of:

1. Generating predictions for overlapping temporal windows.
2. Reconstructing a continuous signal by averaging overlapping predictions.
3. Integrating the predicted differential waveform to obtain the BVP signal.
4. Applying identical postprocessing to predicted and reference waveforms.
5. Estimating heart rate using FFT in the frequency range of 0.5–3.5 Hz.

Model performance is assessed using Pearson correlation and heart-rate estimation errors (MAE).

## Cross-Dataset Evaluation

| Model | Evaluation dataset | Validation subjects | Pearson | HR MAE | HR RMSE |
|---|---|---:|---:|---:|---:|
| PhysNet baseline | UBFC-rPPG | 42 | 0.750 | 1.63 | 2.64 |
| PhysNet (fine-tuned) | MCD-rPPG | 60 | 0.822 | 1.58 | 3.03 |

## Evaluation on a Private Dataset

The fine-tuned PhysNet model was evaluated on an independent private dataset containing recordings not used during training.

| Metric | Value |
|---|---:|
| Mean Pearson correlation | 0.9015 |
| MAE of mean HR estimates | 0.463 BPM |
| RMSE of mean HR estimates | 0.724 BPM |
| Evaluated recordings | 8 |

## Limitations

An additional experiment was conducted using recordings from a separate subject before and after physical exercise.

| Metric | Before exercise | After exercise |
|---|---:|---:|
| Mean Pearson correlation | 0.8695 | 0.4863 |
| Absolute mean HR error (BPM) | 0.279 | 15.780 |
| Predicted mean HR | 80.020 | 106.308 |
| Reference mean HR | 79.741 | 122.088 |

After physical exercise, the model demonstrated a substantial decrease in waveform correlation and an increase in heart-rate estimation error.

## Model Weights

The pretrained weights are provided in `physnet.pth`.

Download using Hugging Face CLI:

```bash
hf download jakuzya/rppg-physnet physnet.pth --local-dir .
```


## References

Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2017).
Unsupervised skin tissue segmentation for remote photoplethysmography.
Pattern Recognition Letters.
Dataset: https://sites.google.com/view/ybenezeth/ubfcrppg

Egorov, K., Botman, S., Blinov, P., Zubkova, G., Ivaschenko, A., Kolsanov, A., & Savchenko, A. (2025).
Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation.
Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM 2025).
DOI: https://doi.org/10.1145/3746027.3758255
Dataset: https://huggingface.co/datasets/kyegorov/mcd_rppg

