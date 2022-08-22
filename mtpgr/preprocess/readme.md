# The dataset production instruction.

# TODO: This file needs updates

Follow this instruction to record and annotate a custom dataset. 

1. Record the gesture video.
2. Download `Losslesscut` as the annotation tool.
3. Annotate the video with `Losslesscut`, mark the segments and class names. The output `.llc` file is in json format, e.g.:

Suffix: **.llc**
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

4. Run `1_` to convert to per-frame class label with `opencv-python`. 

```json
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
// suffix: .json
```

5. Run `2_` to split video frames to individual images in a folder named `<videoname>.images`


6. Run `3_` to generate ground-truth tracking results for police with `multi-person-tracker`.

```json
{5: {'bbox': ndarray, 'frames':  ndarray}, 6: {...}}
// suffix: .pkl
```
7. Tracker can not recover from occlusion, which results in multiple tracks for same person. Run `4_` to concat the tracks for police with non-maximum suppression. 


```json
{1: {'bbox': ndarray, 'frames':  ndarray}}  
// suffix: .pkl
```
number 1 is the person_id of police. other tracks are deleted.
8. Run `5_` to convert tracks to vibe. 
   
Suffix: **.vibe**