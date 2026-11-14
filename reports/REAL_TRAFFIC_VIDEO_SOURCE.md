# Real traffic video and review frames

The app's example is a real drone recording of a four-way **signalized intersection with dedicated right-turn lanes**, from the FLUID research dataset. It is not a roundabout, simulation, generated picture, or image montage.

- Dataset: [FLUID, version 2](https://figshare.com/articles/dataset/29974954/2), DOI `10.6084/m9.figshare.29974954.v2`.
- Authors: Yiyang Chen, Zhigang Wu, Guohong Zheng, Xuesong Wu, Liwen Xu, Haoyuan Tang, Zhaocheng He, and Haipeng Zeng.
- Paper: [A Fine-Grained Lightweight Urban Signalized-Intersection Dataset of Dense Conflict Trajectories](https://www.nature.com/articles/s41597-026-07110-9), Scientific Data (2026).
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution, license link, and the description of changes when sharing the clip or extracted pictures. The authors do not endorse this app.
- Recording: `FIDRT/20250525_video.mp4`, 407.4 seconds in the released source. The included excerpt is seconds **0–60**, at 10 frames per second, resized from 3840×2160 to 1280×720 and encoded as H.264. The source authors stabilized the footage and reduced its frame rate. This app adds no generated traffic or objects.
- Pictures: **60 real frames**, source times 0, 1, …, 59 seconds (source indices 0, 10, …, 590). They are unreviewed examples, not ground-truth preferred actions.

## Reproduce the local sample

Media are kept in ignored `data/signalized_traffic/`, not committed to Git. Install FFmpeg or the optional Python package `imageio-ffmpeg`, then run:

```text
python scripts/setup_signalized_video.py
```

An existing FFmpeg can be supplied using `--ffmpeg PATH`. The installer uses immutable Figshare file IDs, requests only 94 MB of video plus small metadata ranges, checks exact range responses and SHA-256 values, and aborts if the source changes or a server tries to return a complete multi-gigabyte archive. It needs about 400 MB of temporary free space. It refuses to overwrite an existing manifest so reviewed work is preserved.

The outputs are a roughly 11 MB video, 60 JPEG pictures, a manifest with source frame numbers and hashes, an attribution file, and the original observed signal CSV. Every frame has recording group `FLUID:FIDRT:20250525`. Keep that complete group together when splitting train, calibration, and test data.

With the app running, `python scripts/import_signalized_frames.py` analyzes the 60 verified pictures and fills the saved review gallery. This runs the real local models and leaves every human label unset. Repeating the command refreshes model outputs while preserving existing reviews. Alternatively, use **Collect spaced pictures** in the video monitor.

## What the signals mean

The original `20250525_video_signal.csv` records actual signal states by direction and movement, synchronized to the video in seconds. Its timestamp origin is the start of the source recording; the excerpt starts at the same origin. In the source CSV, `s` means straight, `l` means left/U-turn, `p` means pedestrian; `G`, `y`, and `r` mean green, yellow, and red.

For example, the source reports north/south left-turn greens from 0 to 32 seconds, yellow from 32 to 35 seconds, and east/west straight greens from 35 to 60 seconds. These are **observations of the existing controller**. They do not establish that its action was optimal, that another action would be safe, or that this app's proposal is correct. The app must keep source-observed signals distinct from its own proposed action.

The complete source intersection uses protected turns and dedicated slip lanes. A simplified north–south/east–west classroom proposal does not model every permitted movement and should be presented as an observation-only explanation.

## What this adds to training

The clip supplies real pixels that the pretrained detector can process and that a person can review. No automatic preferred-action label, anomaly label, or accuracy claim is created from the current signal or from the detector's own prediction. The user can compare predictions with the visible scene, supply an explicit normal/anomaly/uncertain review, and separately label a preferred light action when the scene supports one. An uncertain review supplies no binary training label. The app does not yet retrain the anomaly network from these new reviews; exported reviews are preparation for a later supervised run.

One minute from one recording is a useful demonstration and one training group, **not independent validation**. Adjacent frames are highly correlated. Claims such as 90% reliability need additional independently recorded signalized intersections, reviewed annotations, held-out evaluation, and uncertainty reporting. Tiny aerial pedestrians, occlusion, and queue tails outside the frame can remain difficult for the current general-purpose pretrained detector.

The video monitor does not connect to, command, or change any real traffic light.
