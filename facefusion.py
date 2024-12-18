#!/usr/bin/env python3
from facefusion import core
import warnings

import os

os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

if __name__ == '__main__':
	core.cli()
