install:
	python -m pip install -U pip
	python -m pip install -e .

install-contribute:
	python -m pip install -U pip
	python -m pip install -r requirements-cont.txt

install-test:
	python -m pip install -U pip
	python -m pip install -r requirements-test.txt