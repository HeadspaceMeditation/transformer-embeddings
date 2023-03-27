# Transformer Embeddings

[![PyPI](https://img.shields.io/pypi/v/transformer-embeddings.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/transformer-embeddings.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/transformer-embeddings)][python version]
[![License](https://img.shields.io/pypi/l/transformer-embeddings)][license]

[![Tests](https://github.com/ginger-io/transformer-embeddings/workflows/Tests/badge.svg?branch=main)][tests]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/transformer-embeddings/
[status]: https://pypi.org/project/transformer-embeddings/
[python version]: https://pypi.org/project/transformer-embeddings
[read the docs]: https://transformer-embeddings.readthedocs.io/
[tests]: https://github.com/ginger-io/transformer-embeddings/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/ginger-io/transformer-embeddings
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

This library simplifies and streamlines the usage of encoder transformer models supported by [HuggingFace's `transformers` library](https://github.com/huggingface/transformers/) ([model hub](https://huggingface.co/models) or local) to generate embeddings for string inputs, similar to the way `sentence-transformers` does.

Please note that starting with v4, we have dropped support for Python 3.7. If you need to use this library with Python 3.7, the latest compatible release is [`version 3.1.0`](https://pypi.org/project/transformer-embeddings/3.1.0/).

## Why use this over HuggingFace's `transformers` or `sentence-transformers`?

Under the hood, we take care of:

1. Can be used with any model on the HF model hub, with sensible defaults for inference.
2. Setting the PyTorch model to `eval` mode.
3. Using `no_grad()` when doing the forward pass.
4. Batching, and returning back output in the format produced by HF transformers.
5. Padding / truncating to model defaults.
6. Moving to and from GPUs if available.

## Installation

You can install _Transformer Embeddings_ via [pip] from [PyPI]:

```console
$ pip install transformer-embeddings
```

## Usage

```python
from transformer_embeddings import TransformerEmbeddings

transformer = TransformerEmbeddings("model_name")
```

If you have a previously instantiated `model` and / or `tokenizer`, you can pass that in.

```python
transformer = TransformerEmbeddings(model=model, tokenizer=tokenizer)
```

```python
transformer = TransformerEmbeddings(model_name="model_name", model=model)
```

or

```python
transformer = TransformerEmbeddings(model_name="model_name", tokenizer=tokenizer)
```

**Note:** The `model_name` should be included if only 1 of model or tokenizer are passed in.

### Embeddings

To get output embeddings:

```python
embeddings = transformer.encode(["Lorem ipsum dolor sit amet",
                                 "consectetur adipiscing elit",
                                 "sed do eiusmod tempor incididunt",
                                 "ut labore et dolore magna aliqua."])
embeddings.output
```

### Pooled Output

To get pooled outputs:

```python
from transformer_embeddings import TransformerEmbeddings, mean_pooling

transformer = TransformerEmbeddings("model_name", return_output=False, pooling_fn=mean_pooling)

embeddings = transformer.encode(["Lorem ipsum dolor sit amet",
                                "consectetur adipiscing elit",
                                "sed do eiusmod tempor incididunt",
                                "ut labore et dolore magna aliqua."])

embeddings.pooled
```

### Exporting the Model

Once you are done testing and training the model, it can be exported into a single tarball:

```python
from transformer_embeddings import TransformerEmbeddings

transformer = TransformerEmbeddings("model_name")
transformer.export(additional_files=["/path/to/other/files/to/include/in/tarball.pickle"])
```

This tarball can also be uploaded to S3, but requires installing the S3 extras (`pip install transformer-embeddings[s3]`). And then using:

```python
from transformer_embeddings import TransformerEmbeddings

transformer = TransformerEmbeddings("model_name")
transformer.export(
    additional_files=["/path/to/other/files/to/include/in/tarball.pickle"],
    s3_path="s3://bucket/models/model-name/date-version/",
)
```

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [Apache 2.0 license][license], _Transformer Embeddings_ is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Credits

This project was partly generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/ginger-io/transformer-embeddings/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/ginger-io/transformer-embeddings/blob/main/LICENSE
[contributor guide]: https://github.com/ginger-io/transformer-embeddings/blob/main/CONTRIBUTING.md
[command-line reference]: https://transformer-embeddings.readthedocs.io/en/latest/usage.html
