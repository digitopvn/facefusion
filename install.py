#!/usr/bin/env python3

from facefusion import installer
import os

os.environ['SYSTEM_VERSION_COMPAT'] = '0'


if __name__ == '__main__':
	installer.cli()
