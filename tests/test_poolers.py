from random import randint

from pytest import mark
from torch import Tensor, equal, mean, ones, rand
from transformers import BatchEncoding
from transformers.file_utils import ModelOutput

from transformer_embeddings.poolers import get_pooler_output, mean_pooling


@mark.repeat(10)
def test_get_pooler_output():
    # Use a random vector with a random batch size but 768 dimensions.
    batch_size = randint(1, 100)
    model_output = ModelOutput(pooler_output=rand(batch_size, 768))
    result = get_pooler_output(model_output=model_output, model_inputs=None)
    assert isinstance(result, Tensor)
    assert equal(result, model_output.pooler_output)


@mark.repeat(10)
def test_mean_pooling():
    tokens = randint(1, 100)
    # Single batch.
    # `BatchEncoding` objects are created with dicts as the first param.
    model_input = BatchEncoding({"attention_mask": ones(1, tokens)})
    model_output = ModelOutput(last_hidden_state=rand(1, tokens, 768))

    mean_pooled = mean_pooling(model_output, model_input)
    assert isinstance(mean_pooled, Tensor)
    # We take mean on the sequence dimension (1).
    assert equal(mean_pooled, mean(model_output.last_hidden_state, dim=1))
