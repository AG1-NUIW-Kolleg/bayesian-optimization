![maintained](https://img.shields.io/static/v1?label=Maintained%3F&message=yes&color=brightgreen)
![made with Python](https://img.shields.io/static/v1?label=made%20with&message=Python&color=007396&logo=Python)
# Project Title
Bayesian Optimization for muscle models.

## Description
This repository applies Bayesian Optimization on muscle models.

## Getting Started

### Dependencies

> This `README.md` only covers the setup of using `Linux`. If you are using `Windows` only or `macOS`, feel free to update the `README.md` with the necessary steps.

Install the newest version of [Python](https://www.python.org/downloads/)

For creating a virtual environment, install [virtualenv](https://virtualenv.pypa.io/en/latest/) with:

```cmd
sudo apt-get install python-virtualenv
```

or

```cmd
pip install virtualenv
```

### Installing

First, clone the repository by running:
```cmd
git clone https://github.com/veit1337/bayesian-optimization.git
```

If there are submodules present, initialize them with:
```cmd
git submodule update --init
```

After that, create a virtual environment with:
```cmd
python -m venv .venv
```

Active the virtual environment by running:
```cmd
source .venv/bin/activate
```

Then, install the dependencies with:
```cmd
pip install -r requirements.txt
```

### Executing program

Run the `main.py` file from root with:
```cmd
python main.py
```

### Contributing

Use [pre-commit hooks](https://pre-commit.com/) for contributions. To active the pre-commit hooks, run:
```cmd
pre-commit install
```

If you add new dependencies, add them to the requirements by running:
```cmd
pip freeze > requirements.txt
```

## Help

For any advise for common problems or issues contact the authors.

## Authors

AUTHORS
