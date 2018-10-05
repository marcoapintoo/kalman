import os
import sys
import pytest
sys.path.append(
    os.path.join(
        os.path.realpath(os.path.dirname(sys.argv[0])), "../../../src/python/pure/"
    )
)
pytest.main(os.path.realpath(sys.path[-1] + "/kalman.py"))
