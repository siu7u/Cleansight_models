"""让 ``pytest framework/tests`` 能直接 import cleansight_eval。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
