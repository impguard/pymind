.PHONY: venv, check, really-check

default:

check:
	nosetests

really-check:
	nosetests --with-coverage --cover-erase --cover-html --cover-package=pymind

clean:
	rm -f $(shell find pymind -name *.pyc)
	rm -f $(shell find test -name *.pyc)
