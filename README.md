# Markovflow

<!-- TODO: -->
<!-- [![PyPI version](https://badge.fury.io/py/markovflow.svg)](https://badge.fury.io/py/markovflow) -->
<!-- [![Coverage Status](https://codecov.io/gh/secondmind-labs/markovflow/branch/develop/graph/badge.svg?token=<token>)](https://codecov.io/gh/secondmind-labs/markovflow) -->
[![Quality checks and Tests](https://github.com/secondmind-labs/markovflow/actions/workflows/quality-check.yaml/badge.svg)](https://github.com/secondmind-labs/markovflow/actions/workflows/quality-check.yaml)
[![Docs build](https://github.com/secondmind-labs/markovflow/actions/workflows/deploy.yaml/badge.svg)](https://github.com/secondmind-labs/markovflow/actions/workflows/deploy.yaml)

[Documentation](https://secondmind-labs.github.io/markovflow/) |
[Tutorials](https://secondmind-labs.github.io/markovflow/tutorials.html) |
[Slack](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw)

## What does Markovflow do?

Markovflow is a Python library for time-series analysis via probabilistic inference in generative models with Markovian Gaussian Processes.

Markovflow uses the mathematical building blocks from [GPflow](http://www.gpflow.org/) and marries these with objects and operators dedicated to run inference and learning in linear dynamical systems. This combination leads to a framework that can be used for:

- researching new Markovian Gaussian process based models, and
- building, training, evaluating and deploying Gaussian processes in a modern way — making use of the tools developed by the deep learning community.


## Getting started

In the [Documentation](https://secondmind-labs.github.io/markovflow/), we have multiple [Tutorials](https://secondmind-labs.github.io/markovflow/tutorials.html) showing the basic functionality of the toolbox.


## Install Markovflow

This project is assuming you are using `python3`.

#### For users

To install the latest (stable) release of the toolbox from [PyPI](https://pypi.org/), use `pip`:
```bash
$ pip install markovflow
```
#### For contributors

To install this project in editable mode, run the commands below from the root directory of the `markovflow` repository.
```bash
poetry install
```
Check that the installation was successful by running the tests:
```bash
poetry run task test
```
You can have a peek at the [Makefile](Makefile) for the commands.


## The Secondmind Labs Community

### Getting help

**Bugs, feature requests, pain points, annoying design quirks, etc:**
Please use [GitHub issues](https://github.com/secondmind-labs/markovflow/issues/) to flag up bugs/issues/pain points, suggest new features, and discuss anything else related to the use of Markovflow that in some sense involves changing the Markovflow code itself. We positively welcome comments or concerns about usability, and suggestions for changes at any level of design. We aim to respond to issues promptly, but if you believe we may have forgotten about an issue, please feel free to add another comment to remind us.

### Slack workspace

We have a public [Secondmind Labs slack workspace](https://secondmind-labs.slack.com/). Please use this [invite link](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw) and join the #markovflow channel, whether you'd just like to ask short informal questions or want to be involved in the discussion and future development of Markovflow.


### Contributing

All constructive input is very much welcome. For detailed information, see the [guidelines for contributors](CONTRIBUTING.md).


### Maintainers

Markovflow was originally created at [Secondmind Labs](https://www.secondmind.ai/labs/) and is now maintained by (in alphabetical order)
[Vincent Adam](https://vincentadam87.github.io/),
[Stefanos Eleftheriadis](https://stefanosele.github.io/),
[Samuel Willis](https://uk.linkedin.com/in/samuel-j-willis).
**We are grateful to [all contributors](CONTRIBUTORS.md) who have helped shape Markovflow.**

Markovflow is an open source project. If you have relevant skills and are interested in contributing then please do contact us (see ["The Secondmind Labs Community" section](#the-secondmind-labs-community) above).

We are very grateful to our Secondmind Labs colleagues, maintainers of [GPflow](https://github.com/GPflow/GPflow), [GPflux](https://github.com/secondmind-labs/GPflux), [Trieste](https://github.com/secondmind-labs/trieste) and [Bellman](https://github.com/Bellman-devs/bellman), for their help with creating contributing guidelines, instructions for users and open-sourcing in general.


## License

[Apache License 2.0](LICENSE)
