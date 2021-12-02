import cv2  # otherwise the MPT reports segment fault
from multi_person_tracker import MPT
from pathlib import Path
import torch
import pickle


def to_tracking(image_folder: Path):
    """
    A000.images -> A000.tracking
    :param image_folder:
    :return:
    """

    assert image_folder.is_dir()
    assert image_folder.suffix == '.images'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mot = MPT(
        device=device,
        batch_size=5,
        display=False,
        detector_type='yolo',
        output_format='dict',
    )
    tracking_results = mot(image_folder)

    save_path = image_folder.parent / (image_folder.stem + '.track')
    with save_path.open('wb') as f:
        pickle.dump(tracking_results, f)



if __name__ == '__main__':
    for img_folder in Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/').glob('*.images'):
        to_tracking(img_folder)
    # image_folder = Path('/home/zc/文档/3. PGv2标注和错误动作剪辑/4K9A0217.images')
    # to_tracking(image_folder)
