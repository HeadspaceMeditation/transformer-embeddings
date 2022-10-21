from pathlib import Path
from tarfile import is_tarfile
from tarfile import open as tarfile_open
from typing import Callable

from pytest import fixture, raises
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.file_utils import ModelOutput

from transformer_embeddings import TransformerEmbeddings
from transformer_embeddings.model import (
    TransformerEmbeddingsOutput,
    TransformerInputOutput,
)
from transformer_embeddings.poolers import mean_pooling


MESSAGES = [
    "Lorem ipsum dolor sit amet",
    "consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
]


@fixture(
    params=[
        "prajjwal1/bert-tiny",
        "sshleifer/tiny-distilroberta-base",
        # "patrickvonplaten/longformer-random-tiny": We cannot test this since this model doesn't have a tokenizer.
        # https://huggingface.co/patrickvonplaten/longformer-random-tiny/
    ]
)
def model_name(request) -> str:
    return request.param


@fixture(params=[True, False])
def return_input(request) -> bool:
    return request.param


@fixture(params=[True, False])
def return_output(request) -> bool:
    return request.param


@fixture(params=[len(MESSAGES), 1])
def batch_size(request) -> int:
    return request.param


# 2nd pooling_fn simply returns the pooler_output from the model.
@fixture(params=[mean_pooling, lambda x, y: x.pooler_output])
def pooling_fn(request) -> Callable:
    return request.param


def test_transformer_embeddings_model_name(model_name):
    # Test transformer object init from model_name.
    transformer = TransformerEmbeddings(model_name=model_name)

    assert transformer.model is not None
    assert transformer.tokenizer is not None

    assert isinstance(transformer.model, PreTrainedModel)
    assert isinstance(
        transformer.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    )


def test_transformer_embeddings_model_tokenizer(model_name):
    # Test transformer object init from model and tokenizer.
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = TransformerEmbeddings(model=model, tokenizer=tokenizer)

    assert transformer.model is not None
    assert transformer.tokenizer is not None

    assert isinstance(transformer.model, PreTrainedModel)
    assert isinstance(
        transformer.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    )


def test_transformer_embeddings_model(model_name):
    # Should raise ValueError if we pass in a model but no model name or tokenizer.
    model = AutoModel.from_pretrained(model_name)
    with raises(ValueError):
        TransformerEmbeddings(model=model)


def test_transformer_embeddings_tokenizer(model_name):
    # Should raise ValueError if we pass in a tokenizer but no model name or model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with raises(ValueError):
        TransformerEmbeddings(tokenizer=tokenizer)


def test_transformer_embeddings_tokenize(model_name):
    # Test tokenization.
    transformer = TransformerEmbeddings(model_name=model_name)
    tokenized_input = transformer.tokenize(MESSAGES)

    assert isinstance(tokenized_input, BatchEncoding)

    for key, value in tokenized_input.items():
        assert isinstance(key, str)
        assert isinstance(value, Tensor)


def assert_transformer_input_output(transformer_input_output: TransformerInputOutput):
    for key, value in transformer_input_output.items():
        assert isinstance(key, str)
        isinstance(value, Tensor)
        # Forward pass with no_grad() doesn't set requires_grad.
        assert not value.requires_grad
        assert value.size()[0] == len(MESSAGES)


def test_transformer_embeddings_encode(
    model_name, batch_size, return_input, return_output, pooling_fn
):
    # Test embedding generation.
    transformer = TransformerEmbeddings(
        model_name=model_name,
        batch_size=batch_size,
        return_input=return_input,
        return_output=return_output,
        pooling_fn=pooling_fn,
    )
    embeddings_output = transformer.encode(MESSAGES)

    assert isinstance(embeddings_output, TransformerEmbeddingsOutput)

    output, input, pooled = (
        embeddings_output.output,
        embeddings_output.input,
        embeddings_output.pooled,
    )

    assert (output is not None) == transformer.return_output
    assert (input is not None) == transformer.return_input
    assert (pooled is not None) == transformer.return_pooled

    if output is not None:
        assert isinstance(output, ModelOutput)
        assert_transformer_input_output(output)

    if input is not None:
        assert isinstance(input, BatchEncoding)
        assert_transformer_input_output(input)

    if pooled is not None:
        assert isinstance(pooled, Tensor)
        assert pooled.size()[0] == len(MESSAGES)


def test_transformer_embeddings_export(model_name, tmp_path):
    transformer = TransformerEmbeddings(model_name=model_name)
    compressed_file = transformer.export(output_dir=tmp_path)

    assert isinstance(compressed_file, Path)
    assert compressed_file.exists()
    assert is_tarfile(compressed_file)


def test_transformer_embeddings_export_additional_files(model_name, tmp_path):
    # Additional file.
    single_file = tmp_path.joinpath("latest")
    single_file.write_text("this is a random text file.")

    transformer = TransformerEmbeddings(model_name=model_name)
    compressed_file = transformer.export(
        output_dir=tmp_path.joinpath("compressed"),
        additional_files=[single_file.as_posix()],
    )

    assert isinstance(compressed_file, Path)
    assert compressed_file.exists()
    assert is_tarfile(compressed_file)

    tar_file = tarfile_open(compressed_file)
    assert single_file.name in tar_file.getnames()
