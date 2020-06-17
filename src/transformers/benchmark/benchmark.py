# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Benchmarking the library on inference and training in PyTorch.
"""


import logging
import timeit

from transformers import (
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    PretrainedConfig,
    is_py3nvml_available,
    is_torch_available,
)

from .benchmark_utils import (
    Benchmark,
    Memory,
    measure_peak_memory_cpu,
    run_on_separate_process,
    start_memory_tracing,
    stop_memory_tracing,
)


if is_torch_available():
    import torch
    from .benchmark_args import PyTorchBenchmarkArguments


if is_py3nvml_available():
    import py3nvml.py3nvml as nvml


logger = logging.getLogger(__name__)


class PyTorchBenchmark(Benchmark):

    args: PyTorchBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "PyTorch"

    @property
    def framework_version(self):
        return torch.__version__

    @run_on_separate_process
    def inference_speed(self, model_name, batch_size, sequence_length):
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_inference)

    @run_on_separate_process
    def inference_memory(self, model_name, batch_size, sequence_length):
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_inference)

    @run_on_separate_process
    def train_speed(self, model_name, batch_size, sequence_length):
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_train)

    @run_on_separate_process
    def train_memory(self, model_name, batch_size, sequence_length):
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_train)

    def _prepare_inference_func(self, model_name, batch_size, sequence_length):
        config = self.config_dict[model_name]

        if self.args.torchscript:
            config.torchscript = True
        if self.args.with_lm_head:
            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)
        else:
            model = MODEL_MAPPING[config.__class__](config)

        model.eval()
        model.to(self.args.device)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        if self.args.fp16:
            logger.info("Running training in Mixed Precision...")
            assert self.args.is_gpu, "Mixed precision is possible only for GPU."
            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        if self.args.torchscript:
            with torch.no_grad():
                inference_model = torch.jit.trace(model, input_ids)
        else:
            inference_model = model

        def encoder_decoder_forward():
            with torch.no_grad():
                inference_model(input_ids, decoder_input_ids=input_ids)

        def encoder_forward():
            with torch.no_grad():
                inference_model(input_ids)

        _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
        return _forward

    def _prepare_train_func(self, model_name, batch_size, sequence_length):
        config = self.config_dict[model_name]
        model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        if self.args.torchscript:
            raise NotImplementedError("Training for torchscript is currently not implemented")
        else:
            train_model = model

        model.eval()
        model.to(self.args.device)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        if self.args.fp16:
            logger.info("Running training in Mixed Precision...")
            assert self.args.is_gpu, "Mixed precision is possible only for GPU."

            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        def compute_loss_and_backprob_encoder():
            loss = train_model(input_ids, labels=input_ids)[0]
            loss.backward()
            train_model.zero_grad()

        def compute_loss_and_backprob_encoder_decoder():
            loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
            loss.backward()
            train_model.zero_grad()

        _train = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        return _train

    def _measure_speed(self, func):
        try:
            if self.args.is_tpu or self.args.torchscript:
                # run additional 10 times to stabilize compilation for tpu and torchscript
                logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                timeit.repeat(
                    func, repeat=1, number=5,
                )

            # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
            runtimes = timeit.repeat(func, repeat=self.args.repeat, number=10,)

            if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
                import torch_xla.debug.metrics as met

                self.print_fn(met.metrics_report())

            return min(runtimes) / 10.0
        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            return "N/A"

    def _measure_memory(self, func):
        try:
            if self.args.trace_memory_line_by_line:
                trace = start_memory_tracing("transformers")

            if self.args.is_tpu:
                # tpu
                raise NotImplementedError(
                    "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.no_memory=True`"
                )
            elif self.args.is_gpu:
                if not is_py3nvml_available():
                    logger.warning(
                        "py3nvml not installed, we won't log GPU memory usage. "
                        "Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                    memory = "N/A"
                else:
                    # init nvml
                    nvml.nvmlInit()
                    func()
                    handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                    meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                    max_bytes_in_use = meminfo.used
                    memory = Memory(max_bytes_in_use)
                    # shutdown nvml
                    nvml.nvmlShutdown()
            else:
                # cpu
                memory_bytes = measure_peak_memory_cpu(func)
                memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

            if self.args.trace_memory_line_by_line:
                summary = stop_memory_tracing(trace)
            else:
                summary = None

            return memory, summary
        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            return "N/A", None
