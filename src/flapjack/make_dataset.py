import random
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd







for idx in valid_ids:
    rgb = load_rgb(idx, data)
    red = load_red(idx, data)
    H, W = red.shape
    rgb = cv2.resize(rgb, (W, H), cv2.INTER_LINEAR)

    n = 0
    while n < images_per_survey:
        x, y = crop_misaligned(
            rgb,
            red,
            sz_range=sz_range,
            max_angle=max_angle,
            max_shift=max_shift,
            normalize=True,
        )

        if (x[..., 0] == 0).mean() > 0.5:
            continue
        else:
            ts = int(time.time() * 1000)
            np.save(data / f"{folder}/valid/images/image_{idx}_{ts}.npy", x)
            np.save(data / f"{folder}/valid/targets/target_{idx}_{ts}.npy", y)
            n += 1
