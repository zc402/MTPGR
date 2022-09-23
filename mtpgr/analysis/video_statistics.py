import datetime
import json
from pathlib import Path
from mtpgr.config import get_cfg_defaults
import cv2
import matplotlib.pyplot as plt
import numpy as np

cfg = get_cfg_defaults()

def video_duration():
    videos = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    videos = videos.glob('*.m4v')

    result = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))

        # count the number of frames
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # calculate duration of the video
        seconds = round(frames / fps)
        # video_time = datetime.timedelta(seconds=seconds)
        result.append((video.stem, seconds))
    print(result)
    print(f'Sum:{sum(list(zip(*result))[1])}')
    print(f'Max:{max(result, key=lambda x:x[1])}')
    print(f'Min:{min(result, key=lambda x:x[1])}')

def clip_duration():

    assert Path(cfg.DATA_ROOT).is_dir(), 'MTPGR/data not found. Expecting "./MTPGR" as working directory'
    combine_folder: Path = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.GENDATA.COMBINE_LABEL_DIR
    video_folder = Path(cfg.DATA_ROOT) / cfg.DATASET.PGDS2_DIR / cfg.DATASET.VIDEO_DIR
    video_paths = video_folder.glob('*.m4v')
    video_paths = list(video_paths)

    collect = []

    for video in video_paths:
        video_name = video.stem
        combine_file = combine_folder / (video_name + ".json")

        with open(combine_file, 'r') as f:
            combine_list = json.load(f)
        collect.extend(combine_list)

    category_names = ['Self', 'Left',
                  'Opposite', 'Right']

    fc = {}  # Frame count
    for i in range(1, 33):
        fc[i] = len([a for a in collect if a == i]) / 25
    results = {
        'Stop':                 [fc[1], fc[9], fc[17], fc[25]],
        'Forward':              [fc[2], fc[10], fc[18], fc[26]],
        'Left turn':            [fc[3], fc[11], fc[19], fc[27]],
        'LT waiting':           [fc[4], fc[12], fc[20], fc[28]],
        'Right turn':           [fc[5], fc[13], fc[21], fc[29]],
        'Lane changing':        [fc[6], fc[14], fc[22], fc[30]],
        'Slow down':            [fc[7], fc[15], fc[23], fc[31]],
        'Pull over':            [fc[8], fc[16], fc[24], fc[32]],
    }


    def survey(results, category_names):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
        """
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.colormaps['viridis'](
            np.linspace(0.15, 0.85, data.shape[1]))

        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                loc='lower left', fontsize='small')

        return fig, ax

    survey(results, category_names)
    plt.savefig('frame_cnt.pdf')

clip_duration()