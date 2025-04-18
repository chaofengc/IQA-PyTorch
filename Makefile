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
test_main: test_forward test_cal
test: test_forward test_cal test_cs test_grad test_dataset 

test_cal:
	pytest tests/ -m calibration -v

test_forward:
	pytest tests/test_metric_general.py::test_forward -v  

test_cs:
	pytest tests/test_metric_general.py::test_cpu_gpu_consistency -v

test_grad:
	pytest tests/test_metric_general.py::test_gradient_backward -v

test_dataset:
	pytest tests/test_datasets_general.py::test_datasets_loading -v

test_fr_dataset:
	pytest tests/test_datasets_general.py::test_fr_datasets_loading -v

test_all:
	pytest tests/ -v

clean:
	rm -rf __pycache__
	rm -rf pyiqa/__pycache__
	rm -r pyiqa.egg-info
	rm -rf build
	rm -rf dist
	pip uninstall -y pyiqa || true
