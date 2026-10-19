# MarmoVR Tracking

> [!IMPORTANT]
> **Select the environment and model settings before running the application.**
>
> 1. **Marmoset EfficientNet model:** run in the **`DLC-live`** environment, select the appropriate **`neuralNetDir`**, and set **`modelType = 'base'`**.
> 2. **Puppet ResNet-50 model:** run in the **`DLC-live3`** environment, select the appropriate **`neuralNetDir`**, and set **`modelType = 'pytorch'`**.
>
> Both settings are defined in `Settings/Detection settings.txt`. Keep only one `neuralNetDir` entry active.

Real-time multi-camera tracking for the MarmoVR virtual reality setup. The application acquires XIMEA camera images, detects the subject, estimates anatomical keypoints with DLC-live, and reconstructs 3D positions for monitoring, recording, and interaction with Unreal Engine.

## Features

- Multi-camera acquisition with per-camera monitoring.
- Background-subtraction detection and cropped-image pose estimation.
- TensorFlow and PyTorch inference through DLC-live.
- Dynamic cropping based on detected keypoints.
- Calibrated triangulation of subject positions and anatomical keypoints.
- A configurable “cyclop” position, computed from selected keypoints, for tracking and VR feedback.
- Optional velocity, motion direction, and gaze-direction calculations.
- A PyQt5 interface for experiment and trial control.
- TCP commands and UDP position exchange with the rendering computer.
- Trial video, tracking data, and event recording; optional audio and Recobox control.

## Project files

| Path | Purpose |
| --- | --- |
| `Tracking.py` | Application entry point, acquisition, inference, triangulation, networking, recording, and GUI. |
| `Settings/System settings.txt` | Controller, processing switches, monitoring, recording, and output paths. |
| `Settings/Camera settings.txt` | Camera selection, serial numbers, frame rate, exposure, gain, and crop size. |
| `Settings/Detection settings.txt` | Model path and backend, keypoints, confidence thresholds, cropping, and detector settings. |
| `Settings/camCombUse.txt` | Camera combinations and reference cameras used for triangulation. |
| `Calibration/` | Camera projection matrices, coordinate transforms, and calibration error reports. |
| `Masks/` | Per-camera exclusion masks. |
| `References/` | Background reference images used by the detector. |

Trained models and the Python environments are not included in this archive. The default model paths point to a sibling `NeuralNetworks` directory.

## Requirements

Use the existing environment corresponding to the selected model. The application depends on:

- Python, NumPy, OpenCV, and PyQt5.
- XIMEA camera drivers/SDK and the `ximea.xiapi` Python interface.
- DLC-live and its matching TensorFlow or PyTorch backend.
- Compatible GPU drivers and runtime libraries when using GPU inference.
- For audio recording: PyAudio, pydub, and MoviePy; audio conversion also requires FFmpeg.

The archive does not provide an environment specification or pinned dependency list. Preserve the working `DLC-live` and `DLC-live3` environments rather than installing both model backends into one environment.

## Model selection and launch

Edit `Settings/Detection settings.txt` before launching. This file uses **tab-separated setting names and Python values**, not assignment statements. Preserve tabs, quotes, and Python boolean syntax when editing.

### Marmoset EfficientNet

Set the following values, adapting the model path to the local installation:

```text
neuralNetDir	'../NeuralNetworks/Marmosets/efficientnet-b0_wide/'
modelType	'base'
```

Activate the corresponding environment and start the application from the project folder:

```bash
conda activate DLC-live
python Tracking.py
```

### Puppet ResNet-50

Set the following values, adapting the checkpoint path to the local installation:

```text
neuralNetDir	'../NeuralNetworks/Puppets/DLC_marmo_puppet_resnet_50_iteration-0_shuffle-1_snapshot-best-190.pt'
modelType	'pytorch'
```

```bash
conda activate DLC-live3
python Tracking.py
```

The puppet example points to a `.pt` checkpoint even though the setting is named `neuralNetDir`. In both cases, verify that `keyNames` matches the model's output order. The application resolves relative paths from the directory containing `Tracking.py`.

## Setup configuration

Before the first acquisition:

1. In `Camera settings.txt`, set `camList` and `camSerialNbs` to the connected cameras. The supplied configuration uses cameras 0–3, a processing frame rate of 40 fps, and a 450 × 450 crop. The requested camera frame rate is `framerate × frameStep`.
2. Use calibration matrices, exclusion masks, and background references corresponding to the actual camera arrangement. The supplied files describe a specific setup and must be checked after any camera movement.
3. Check `calibDir` and `resultsDir` in `System settings.txt`. The default results directory is `../Results/`.
4. Choose `controller = 'UI'` for local experiment control or `'UE'` for control from Unreal Engine.
5. Review the `IP` dictionary and communication ports near the top of `Tracking.py`. Even in UI mode, the TCP server uses the configured tracking address; use an address assigned to the computer or enable `runLocal` for local network communication.
6. Set processing and recording options such as `runDetect`, `runDLC`, `triangulate`, `sendPos3D`, and `saveResults` as needed.

The configuration loader evaluates Python expressions. Only load trusted settings files.

## Run an experiment

With `controller = 'UI'`:

1. Launch `Tracking.py` in the selected environment.
2. Enter the experiment, subject, trial, and condition identifiers in the interface.
3. Click **Start Experiment** to start acquisition and monitoring.
4. Inspect the camera views and keypoint overlays. Use **New Reference Image** when an updated background is needed, with the subject outside the detection area.
5. Enable **Save results** and click **Start Trial** to record a trial.
6. Click **End Trial** and allow the data to finish saving before starting another trial or closing the application.
7. Click **End Experiment** to stop acquisition, then quit the application.

Use distinct trial and condition identifiers to avoid reusing output filenames.

## Network integration

| Connection | Default configuration |
| --- | --- |
| Tracking computer | `192.168.0.2` |
| Rendering computer | `192.168.0.1` |
| Tracking TCP command server | Port `65432` |
| UDP positions sent to rendering | Port `50771` |
| UDP events received from rendering | Port `50772` |
| Recobox 0 / 1 | `192.168.0.20:2390` / `192.168.0.21:2391` |

Set `runLocal = True` in the system settings to use localhost for tracking/rendering communication. This does not simulate cameras or remove hardware dependencies. Recobox addresses remain separately configured in `Tracking.py`.

## Recorded outputs

Trial files are written under `resultsDir/<expID>/<subjectID>/` with the prefix `Trial-<trialID>_Cond-<condID>`.

Depending on the enabled options, outputs include:

- Per-camera `.mp4` videos and copied background reference images.
- Structured NumPy `.npy` arrays containing 2D positions, keypoint confidence values, triangulated 3D positions, and events.
- Tab-separated `.tsv` copies when `saveTextCopy` is enabled.
- Audio files when `recSound` is enabled.
- An experiment-level `<expID>_files.tsv` index linking identifiers to trial filenames.

Data suffixes indicate the active processing modes, for example `_pos2D+DLC2D`, `_pos3D+DLC3D`, or `_events`. Three-dimensional positions are expressed in the calibrated real-world reference frame.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Model fails to load | Match the environment, `modelType`, and `neuralNetDir` using the banner above. |
| Camera connection fails | Check XIMEA drivers, camera access, serial numbers, and `camList`. |
| GUI does not appear / TCP server cannot bind | Check the tracking IP address and port availability; use `runLocal` when appropriate. |
| Detection or crops are incorrect | Check background references, masks, thresholds, and crop settings. |
| 3D positions are missing or inconsistent | Check calibration files, camera combinations, `triangulate`, and confidence thresholds. |
| Results are missing | Check `saveResults`, the output path, and whether a trial was started and ended. |

Camera triggering and synchronization are listed as TODO items in `Tracking.py`; frame-accurate synchronization should not be assumed from this application alone. The saving code also calls `os.sync()`, which requires platform compatibility checks when moving the application to Windows.
