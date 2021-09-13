# Contribution guidelines

### Who are we?

*Maintainers* (Samuel Willis, Vincent Adam, Stefanos Eleftheriadis) steer the project, keep the community thriving, and manage contributions.

*Contributors* (you?) submit issues, make pull requests, answer questions on Slack, and more.

Community is important to us, and we want everyone to feel welcome and be able to contribute to their fullest. Our [code of conduct](CODE_OF_CONDUCT.md) gives an overview of what that means.

### Reporting a bug

Finding and fixing bugs helps us provide robust functionality to all users. You can either submit a bug report or, if you know how to fix the bug yourself, you can submit a bug fix. We gladly welcome either, but a fix is likely to be released sooner, simply because others may not have time to quickly implement a fix themselves. If you're interested in implementing it, but would like help in doing so, ask in the [community Slack workspace](https://secondmind-labs.slack.com).

We use GitHub issues for bug reports. You can use the [issue template](https://github.com/secondmind-labs/markovflow/issues/new?assignees=&labels=bug&template=bug_report.md&title=) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible, ideally within the week, and get back to you about how to proceed. If it's a small easy fix, they may implement it then and there. For fixes that are more involved, they will discuss with you about how urgent the fix is, with the aim of providing some timeline of when you can expect to see it.

If you'd like to submit a bug fix, the [pull request templates](https://github.com/secondmind-labs/markovflow/compare) are a good place to start. We recommend you discuss your changes with the community before you begin working on them, so that questions and suggestions can be made early on.

### Requesting a feature

Markovflow is built on features added and improved by the community. You can submit a feature request either as an issue or, if you can implement the change yourself, as a pull request. We gladly welcome either, but a pull request is likely to be released sooner, simply because others may not have time to quickly implement it themselves. If you're interested in implementing it, but would like help in doing so, ask in the [community Slack workspace](https://secondmind-labs.slack.com).

We use GitHub issues for feature requests. You can use the [issue template](https://github.com/secondmind-labs/markovflow/issues/new?assignees=&labels=&template=feature_request.md&title=) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible, ideally within the week, and get back to you about how to proceed. If it's a small easy feature that is backwards compatible, they may implement it then and there. For features that are more involved, they will discuss with you about a timeline for implementing it. Features that are not backwards compatible are likely to take longer to reach a release. It may become apparent during discussions that a feature doesn't lie within the scope of Markovflow, in which case we will discuss alternative options with you, such as adding it as a notebook or an external extension to Markovflow.

If you'd like to submit a pull request, the [pull request templates](https://github.com/secondmind-labs/markovflow/compare) are a good place to start. We recommend you discuss your changes with the community before you begin working on them, so that questions and suggestions can be made early on.

### Pull request guidelines

- Limit the pull request to the smallest useful feature or enhancement, or the smallest change required to fix a bug. This makes it easier for reviewers to understand why each change was made, and makes reviews quicker.
- Where appropriate, include [documentation](#documentation), [type hints](#type-checking), and [tests](#tests). See those sections for more details.
- Pull requests that modify or extend the code should include appropriate tests, or be covered by already existing tests. In particular:
  - New features should include a demonstration of how to use the new API, and should include sufficient tests to give confidence that the feature works as expected.
  - Bug fixes should include tests to verify that the updated code works as expected and defend against future regressions.
  - When refactoring code, verify that existing tests are adequate.
- So that notebook users have the option to import things as
  ```python
  import markovflow
  markovflow.models.SparseVariationalGaussianProcess(...)
  ```
  import all modules (or their contents) in their parent package \_\_init\_\_.py file.
- In commit messages, be descriptive but to the point. Comments such as "further fixes" obscure the more useful information.

### Documentation

Markovflow has two primary sources of documentation: the notebooks and the API reference.

### Quality checks

#### Type checking

We use [type hints](https://docs.python.org/3/library/typing.html) for documentation and static type checking with [mypy](http://mypy-lang.org). We do this throughout the source code and tests. If you don't know how to add type hints to your code, leave them out. You can use `typing.Any` where the actual type isn't expressible or practical, but do avoid it where possible.

#### Tests

We write and run tests with [pytest](https://pytest.org). We aim for all public-facing functionality to have tests for both happy and unhappy paths (that test it works as intended when used as intended, and fails as intended otherwise). We don't test private functionality, as the cost to ease of development is more problematic than the benefit of improved robustness.

Run type checking and tests with
```bash
$ poetry run task test
```

#### Continuous integration

[GitHub actions](https://docs.github.com/en/actions) will automatically run the quality checks against pull requests to develop or master, by calling into tox. The GitHub repository is set up such that these need to pass in order to merge.

### Updating dependencies

To update the Python dependencies used in any part of the project, update `pyproject.toml` and/or any relevant requirements.txt files. Then, in the repository root, run
```bash
$ poetry update
```
This will update the `poetry.lock` file.

# License

[Apache License 2.0](LICENSE)
