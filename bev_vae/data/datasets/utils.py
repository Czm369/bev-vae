import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from pyarrow import feather
from pyquaternion import Quaternion


def read_feather(
    path: Path, columns: Optional[Tuple[str, ...]] = None
) -> pd.DataFrame:

    with path.open("rb") as file_handle:
        dataframe: pd.DataFrame = feather.read_feather(
            file_handle, columns=columns, memory_map=True
        )
    return dataframe


def read_json_file(fpath: Path) -> Dict[str, Any]:
    with fpath.open("rb") as f:
        data: Dict[str, Any] = json.load(f)
        return data

def read_img(img_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def matrix2quaternion(matrix: np.ndarray) -> List[float]:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    # quaternion = Quaternion(matrix=rotation, atol=1e-07)
    quaternion = Quaternion(matrix=rotation, atol=1e-05)
    params = quaternion.q.tolist() + translation.tolist()
    return params

def quaternion2matrix(params: List[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(params[:4]).rotation_matrix
    matrix[:3, 3] = params[4:]
    return matrix.astype(np.float32)
