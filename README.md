# Simple Backtester

A simple, readily extensible backtesting framework for financial time series data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project, `simple_backtester`, is designed to provide a framework for backtesting financial time series data. It is developed by Zhuolin Xiang, Haoqian Zhang, Yuheng Zheng, and Siqing Zou.

## Installation

To install the required dependencies, run:

```sh
pip install .
```

For development, you can install the optional dependencies:

```sh
pip install .[dev]
```

## Usage

To use the backtesting framework, follow these steps:

1. Clone the repository:

```sh
git clone https://github.com/MikamkanSQ123/apc524_proj.git
```

2. Create your own branch:

```sh
git checkout -b your-branch-name
```

3. Add your local configuration files, examples configs are in `tests/test_data/strategy`.

4. To run the whole pipeline, refer to examples in `examples/backtest.ipynb`, or you may choose to interact with the front-end interface.

## Contributing

We welcome contributions to the project. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your branch and create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.