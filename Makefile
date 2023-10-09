refresh: clean build install build_dist release 

build:
	python setup.py build

install:
	python setup.py install

build_dist:
	make clean
	python setup.py sdist bdist_wheel
	pip install dist/*.whl

release:
	python -m twine upload dist/*

# lint:
# 	flake8 pyiqa/ --count --max-line-length=127 --ignore=W293,W503,W504,E126,E741

# test calibration and forward inference
test:
	pytest tests/ -m calibration -v
	pytest tests/test_metric_general.py::test_forward -v  

test_general:
	pytest tests/test_metric_general.py::test_cpu_gpu_consistency -v

test_gradient:
	pytest tests/test_metric_general.py::test_gradient_backward -v

test_dataset:
	pytest tests/test_datasets_general.py -v

test_all:
	pytest tests/ -v

clean:
	rm -rf __pycache__
	rm -rf pyiqa/__pycache__
	rm -r pyiqa.egg-info
	rm -rf build
	rm -rf dist
	pip uninstall -y pyiqa || true
