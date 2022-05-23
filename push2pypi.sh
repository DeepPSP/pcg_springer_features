#!/bin/sh
python setup.py sdist bdist_wheel
twine upload dist/*
rm -rf build dist pcg_springer_features.egg-info
