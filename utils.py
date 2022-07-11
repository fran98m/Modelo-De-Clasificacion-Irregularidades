
import re
import numpy as np
import pandas as pd

def mask_float(value: 'Any') -> np.float64:

    if pd.isnull(value) or str(value) == "":
        return np.float64(0)
    
    else:
        try:
            result = re.search(r"\-*\d*(\,|\.)*\d+", str(value))
            if result is not None:
                found = result.group(0).replace(",", ".")
            else:
                found = 0
            return np.float64(found)

        except ValueError:
            return np.NaN