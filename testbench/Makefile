TESTS = $(wildcard test_*.py)
RUNTESTS = $(TESTS:.py=.test)

default: $(RUNTESTS)

%.test: %.py
	python $<
