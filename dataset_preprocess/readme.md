# The dataset production instruction.

Follow this instruction to record and annotate a custom dataset.

1. Record the gesture video.
2. Download `Losslesscut` as the annotation tool.
3. Annotate the video with `Losslesscut`, mark the segments and class names. The output `.llc` file is in json5 format, e.g.:
```json5
{
  version: 1,
  mediaFileName: '4K9A0217.mp4',
  cutSegments: [
    {
      start: 1.370253,
      end: 3.370233,
      name: '1',
    },
    {
      start: 6.120231,
      end: 8.870245,
      name: '2',
    },
  ],
}
```

4. Convert to per-frame class label with `opencv-python`.
```json5
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
```

5. Split video frames to individual images in a folder named `<videoname>.images`

6. Generate ground-truth tracking results for police with `multi-person-tracker`.