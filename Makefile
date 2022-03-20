refresh: clean build install lint

build:
	python setup.py build

install:
	python setup.py install

build_dist:
	make clean
	python setup.py sdist bdist_wheel
	pip install dist/*.whl
	# make test

release:
	python -m twine upload dist/*

lint:
	flake8 pyiqa/ --count --max-line-length=127 --ignore=W293,W503,W504,E126,E741

# test:
# 	python -m unittest

clean:
	rm -rf __pycache__
	rm -rf pyiqa/__pycache__
	rm -r pyiqa.egg-info
	rm -rf build
	rm -rf dist
	pip uninstall -y pyiqa || true
