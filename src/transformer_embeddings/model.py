from logging import getLogger
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor, cat, cuda, device
from torch.autograd.grad_mode import no_grad
from torch.nn.functional import pad
from tqdm.auto import trange
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.file_utils import ModelOutput

from transformer_embeddings.helpers import compress_files


logger = getLogger(__name__)

MODEL_TARBALL = "model.tar.gz"

DEVICE_CUDA = device("cuda")
DEVICE_CPU = device("cpu")
DEVICE = DEVICE_CUDA if cuda.is_available() else DEVICE_CPU

TransformerInputOutput = Union[BatchEncoding, ModelOutput]


class TransformerEmbeddingsOutput(ModelOutput):
    output: Optional[ModelOutput] = None
    input: Optional[BatchEncoding] = None
    pooled: Optional[Tensor] = None


class TransformerEmbeddings:
    """Thin wrapper on top of the HuggingFace's transformers library to simplify
    generating embeddings."""

    def __init__(
        self,
        model_name: Optional[Union[str, Path]] = None,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        model: PreTrainedModel = None,
        batch_size: int = 16,
        tensors_to_cpu: bool = True,
        return_output: bool = True,
        return_input: bool = False,
        pooling_fn: Callable[[ModelOutput, BatchEncoding], Tensor] = None,
    ):
        """
        Create a TransformerEmbeddings object.

        Parameters
        ----------
        model_name : Union[str, Path], optional
            Name of the model. Anything supported by HF's `from_pretrained()` method
            (HF model hub model name, local path). Default: None.
        tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional
            Tokenizer object. Default: None.
        model : PreTrainedModel, optional
            Model object. Default: None.
        batch_size : int, optional
            Batch size for the foward pass. Default: 16.
        tensors_to_cpu : bool, optional
            Move the output back to CPU if this is operating on GPU? Default: True.
        return_output : bool, optional
            Should all the outputs from the model's forward pass be returned? Default: True.
        return_input : bool, optional
            Should the tokenized inputs be returned? Default: False.
        pooling_fn : Callable[[ModelOutput, BatchEncoding], Tensor], optional
            Function to apply to pool the output produced into a tensor. If provided,
            self.return_pooled is set to True and returned.
        """
        self.load_model(model_name=model_name, tokenizer=tokenizer, model=model)

        self.batch_size = batch_size
        self.tensors_to_cpu = tensors_to_cpu
        self.pooling_fn = pooling_fn

        self.return_output = return_output
        self.return_input = return_input
        self.return_pooled = self.pooling_fn is not None

        logger.info("TransformerEmbedddings model initialized.")

    def load_model(
        self,
        model_name: Optional[str],
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
    ) -> None:
        """
        Load the model.

        Raises
        ------
        ValueError
            If either model or tokenizer are provided without the other, and model_name is also not provided.
        """

        if model_name and tokenizer is None:
            logger.info(f"Loading tokenizer from {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name and model is None:
            logger.info(f"Loading model from {model_name}")
            model = AutoModel.from_pretrained(model_name)

        if tokenizer is None:
            raise ValueError("Tokenizer was not passed or created.")
        else:
            self.tokenizer = tokenizer

        if model is None:
            raise ValueError("Model was not passed or created.")
        else:
            self.model = model.to(DEVICE).eval()
        logger.info(f"Model and tokenizer loaded, on device {DEVICE}, set to eval().")
        if (
            model.config
            and model.config.max_position_embeddings
            and tokenizer.model_max_length
            and model.config.max_position_embeddings != tokenizer.model_max_length
        ):
            logger.warning(
                f"Model's maximum position embeddings ({model.config.max_position_embeddings}) do not match tokenizer's maximum length ({tokenizer.model_max_length})."
            )

    def tokenize(self, input_strings: List[str]) -> BatchEncoding:
        """
        Tokenize the input.

        Parameters
        ----------
        input_strings : List[str]
            Input strings for the batch.

        Returns
        -------
        BatchEncoding
            Tokenizer output.
        """
        return self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    def _pad_tensors(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Pad tensors to equal length."""
        max_length = max(x.shape[1], y.shape[1])
        # padding passed to torch.nn.functional.pad are backwards from the last axis
        # and is specified twice for each dimension (front and back).
        # input tensors are 2D, output tensors are 3D.
        # input and output tensors both have to be padded at dimension 1.
        # padding for input tensors becomes: (0, max_length - x.shape[1])
        # padding for output tensors becomes: (0, 0, 0, max_length - x.shape[1])
        # See: https://pytorch.org/docs/master/generated/torch.nn.functional.pad.html
        # We don't pad along the dimension we stack (dim=0; batch dimension), thus
        # the -2. For the final dimension, we only pad at the back, hence the -1.
        prefix_length = 2 * len(x.shape) - 2 - 1
        pad_prefix = (0,) * prefix_length
        x = pad(x, pad=pad_prefix + (max_length - x.shape[1],))
        y = pad(y, pad=pad_prefix + (max_length - y.shape[1],))
        return x, y

    def _change_device(self, data: TransformerInputOutput) -> TransformerInputOutput:
        """Move tensors to CPU if specified."""
        if self.tensors_to_cpu:
            for key, value in data.items():
                data[key] = value.to(DEVICE_CPU)
        return data

    def _stack_batch(
        self,
        data: Optional[TransformerInputOutput],
        batch_data: TransformerInputOutput,
    ) -> TransformerInputOutput:
        """Stack batch data with data from previous batches."""
        # Stack.
        if data is None:
            data = batch_data
        else:
            for key, value in batch_data.items():
                previous = data[key]
                if previous.shape != value.shape:
                    previous, value = self._pad_tensors(previous, value)
                data[key] = cat((previous, value), dim=0)
        return data

    def _stack_pooled(self, pooled: Optional[Tensor], batch_pooled: Tensor) -> Tensor:
        # Pooling always occurs on tensors on the same device, so we do not need to
        # move them. The pooled tensor would already be on CPU if tensors_to_cpu is
        # True, else they'd be on GPU. Either way, they'd be on the device we want it on.
        return batch_pooled if pooled is None else cat((pooled, batch_pooled), dim=0)

    def encode(self, input_strings: List[str]) -> TransformerEmbeddingsOutput:
        """
        Generate embeddings for the given input.

        Parameters
        ----------
        input_strings : List[str]
            String input for which embeddings should be generated.

        Returns
        -------
        TransformerEmbeddingsOutput
            Model output, inputs and / or pooled output.
        """
        logger.info(f"Generating embeddings for {len(input_strings)} input strings.")
        output = None
        input = None
        pooled = None
        for i in trange(0, len(input_strings), self.batch_size):
            batch_tokenized_input = self.tokenize(
                input_strings[i : i + self.batch_size]
            )

            with no_grad():
                batch_outputs = self.model(**batch_tokenized_input.to(DEVICE))

            # Move all tensors to CPU.
            batch_tokenized_input = self._change_device(batch_tokenized_input)
            batch_outputs = self._change_device(batch_outputs)

            # Stack tensors from batch to output for all inputs.
            if self.return_output:
                output = self._stack_batch(output, batch_outputs)
            if self.return_input:
                input = self._stack_batch(input, batch_tokenized_input)
            if self.return_pooled:
                batch_pooled = self.pooling_fn(batch_outputs, batch_tokenized_input)
                pooled = self._stack_pooled(pooled, batch_pooled)
        return TransformerEmbeddingsOutput(output=output, input=input, pooled=pooled)

    def export(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        additional_files: Optional[List[Union[str, Path]]] = None,
        s3_path: Optional[str] = None,
    ) -> Path:
        """
        Export the model and tokenizer to a directory and compress it into a tarball. If
        an S3 path is provided, also upload the tarball to S3.

        Parameters
        ----------
        output_dir : Optional[Union[str, Path]], optional
            Output directory. Default: A temporary directory is created that is cleaned up when the function returns.
        additional_files : Optional[List[Path]], optional
            Additional files to include in the exported tarball. Default: None.
        s3_path : Optional[str], optional
            S3 path at which to upload the file. Default: None, which means that the file is not uploaded to S3.

        Returns
        -------
        Path
            Tarball path. If it was a temporary directory, it will be empty.
        """
        # Set directory.
        temporary_directory = TemporaryDirectory()
        if output_dir is None:
            output_dir = temporary_directory.name
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        if not output_dir.exists() or output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Copy additional files into output directory before compressing.
        for file in additional_files or []:
            copy2(file, output_dir)

        # Save model, tokenizer.
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Create a tarball.
        logger.debug(f"Folder being added to the tarball is {output_dir}.")
        compressed_file = output_dir.joinpath(MODEL_TARBALL)
        compress_files([output_dir], compressed_file)
        logger.info(f"Tarball {compressed_file} created.")

        if s3_path:
            try:
                from s3fs import S3FileSystem
            except ImportError:
                raise ImportError(
                    "Please install the s3 extras of the package to upload to S3."
                )

            s3_fs = S3FileSystem()
            logger.info(f"Tarball {compressed_file} being uploaded to S3 at {s3_path}.")
            s3_fs.open(s3_path, "wb").write(compressed_file.read_bytes())

        return compressed_file
