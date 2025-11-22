import os
import os.path as osp
from typing import List

import numpy as np


def get_calibration_data(video_path: str) -> List[np.ndarray]:
    """Get calibration data based on the input file."""
    X = [
        [1314, 1422],
        [330, 1192],
        [130, 1387],
        [3013, 1583],
        [1951, 1287],
        [484, 915],
        [2000, 957],
        [2259, 886],
        [2905, 1017],
    ]
    y = [
        309 / 295,
        292 / 280,
        297 / 280,
        328 / 310,
        294 / 310,
        261 / 300,
        255 / 310,
        243 / 310,
        252 / 310,
    ]

    if osp.basename(video_path).endswith(".MTS"):
        X = [
            [742, 722],
            # [330, 1192],
            # [130, 1387],
            [1629, 791],
            [1068, 644],
            [306, 474],
            [1088, 479],
            [1225, 436],
            [1564, 489],
        ]

        y = [
            148 / 295,
            # 292 / 280,
            # 297 / 280,
            171 / 310,
            142 / 310,
            130 / 300,
            136 / 310,
            121 / 310,
            120 / 310,
        ]
    data_calibration = [np.array(X), np.array(y)]
    return data_calibration


def get_homography_data(video_path: str, wgs: bool = False) -> List[np.ndarray]:
    """Get homography data based on the input file."""
    if wgs:
        wgs_points = [
            [[212, 1090], [13.3528189571, 49.7274626145]],
            [[516, 967], [13.3529170517, 49.7274852922]],
            [[1040, 850], [13.3530381870, 49.7274811562]],
            [[1833, 1022], [13.3530644952, 49.7273850415]],
            [[3194, 1277], [13.3530826252, 49.7272308215]],
            [[1545, 1827], [13.3526848010, 49.7272356852]],
        ]
        if osp.basename(video_path).endswith(".MTS"):
            video_points = [
                [141, 564],
                [322, 505],
                [597, 436],
                [1010, 514],
                [1726, 627],
                [857, 935],
            ]
        else:
            video_points = [wgs_point[0] for wgs_point in wgs_points]
        map_points = [wgs_point[1] for wgs_point in wgs_points]

    else:
        map_points = [
            [1573, 274],
            [1887, 159],
            [2319, 162],
            [2410, 672],
            [2458, 1480],
            [1084, 1468],
        ]
        if os.path.basename(video_path) == "C0003_10fps.mp4":
            video_points = [
                [380, 1052],
                [627, 932],
                [1098, 822],
                [1815, 949],
                [3042, 1130],
                [1590, 1668],
            ]
        elif osp.basename(video_path).endswith(".MTS"):
            video_points = [
                [141, 564],
                [322, 505],
                [597, 436],
                [1010, 514],
                [1726, 627],
                [857, 935],
            ]
            map_points = [
                [786, 136],
                [944, 80],
                [2319 / 2, 162 / 2],
                [2410 / 2, 672 / 2],
                [2458 / 2, 1480 / 2],
                [1084 / 2, 1468 / 2],
            ]
        else:
            video_points = [
                [229, 1087],
                [518, 962],
                [1057, 845],
                [1837, 1018],
                [3198, 1276],
                [1559, 1816],
            ]

    video_points = np.array(video_points)
    map_points = np.array(map_points)
    data_homography = [video_points, map_points]
    return data_homography
