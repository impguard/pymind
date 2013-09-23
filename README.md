PyMind
======

Simple Python neural network implementation.

Contributor
===========
Installation
------------
The dependencies for PyMind are located in the requirements.txt file. Note that
using virtualenv is optional, but might be preferable in order to localize the
project.

To get the package simply pull from the repository:

    git clone https://github.com/ImpGuard/PyMind.git <somedir>

In order to install the dependencies, use pip:

    cd <somedir>
    virtualenv <env_name>           # Create virtualenv if desired
    source <evn_name>/bin/activate  # Activate the environment if installed
    pip install -r requirements.txt # Install any necessary modules

Testing
-------
In order to run the tests, use nose:

    cd Pymind
    nosetests

Add coverage tests using the coverage plugin for nose:

    cd Pymind
    nosetests --with-coverage --cover-erase --cover-html --cover-package=pymind

Note: The coverage tests will create a html page detailing coverage in the folder `Pymind/cover/index.html`.

User
====
Installation
------------
Simply pull from the repository:

    git clone https://github.com/ImpGuard/PyMind.git <somedir>

Usage
----
Coming soon!

