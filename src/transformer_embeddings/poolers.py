from torch import Tensor, clamp
from torch import sum as torch_sum
from transformers import BatchEncoding
from transformers.file_utils import ModelOutput


def mean_pooling(model_output: ModelOutput, model_inputs: BatchEncoding) -> Tensor:
    """
    Mean pooling for the model output.

    This is the unweighted average of the token embeddings, while ignoring padding
    and padded tokens.
    Copied from: https://huggingface.co/sentence-transformers/msmarco-distilroberta-base-v2
    Comments ours :).

    Parameters
    ----------
    model_output : ModelOutput
        Output from the model.
    model_inputs : BatchEncoding
        Encoded, tokenized input to the model.

    Returns
    -------
    Tensor
        Mean pooled output.
    """
    # last_hidden_state is the per token embedding.
    last_hidden_state = model_output.last_hidden_state
    attention_mask = model_inputs.attention_mask
    # Expand attention_mask (2D) to the shape of the last_hidden_state (3D).
    attention_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    # Multiply with the expanded attention mask to make the
    # last_hidden_state for padding tokens to 0.
    sum_embeddings = torch_sum(last_hidden_state * attention_mask_expanded, dim=1)
    sum_mask = clamp(attention_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_pooler_output(model_output: ModelOutput, model_inputs: BatchEncoding) -> Tensor:
    """
    Return the pooler output.

    Parameters
    ----------
    model_output : ModelOutput
        Output from the model.
    model_inputs : BatchEncoding
        Encoded, tokenized input to the model. Not used in this function.

    Returns
    -------
    Tensor
        Pooler output.
    """
    return model_output.pooler_output
