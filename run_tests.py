#!/usr/bin/env python

import sys

import nose


if __name__ == '__main__':
    nose_args = sys.argv + ['--config', 'nose.cfg']
    nose.run(argv=nose_args)