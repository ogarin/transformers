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
import os
import random
import timeit
from functools import wraps
from multiprocessing import Process, Queue

from transformers import (
    TF_MODEL_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    PretrainedConfig,
    is_py3nvml_available,
    is_tf_available,
)

from .benchmark_utils import Benchmark, Memory, measure_peak_memory_cpu, start_memory_tracing, stop_memory_tracing


if is_tf_available():
    import tensorflow as tf
    from .benchmark_args_tf import TensorflowBenchmarkArguments
    from tensorflow.python.framework.errors_impl import ResourceExhaustedError

if is_py3nvml_available():
    import py3nvml.py3nvml as nvml

logger = logging.getLogger(__name__)


def run_on_separate_process(func):
    # run function in an individual
    # process to get correct memory
    @wraps(func)
    def process(*args, **kwargs):
        def wrapper_func(queue, *args):
            try:
                logger.info("run with process id: {}".format(os.getpid()))
                result = func(*args)
            except Exception:
                result = "N/A"
                logger.warning("Exception when running on process id {os.getpid()}.")
            queue.put(result)

        queue = Queue()
        p = Process(target=wrapper_func, args=[queue] + list(args))
        p.start()
        result = queue.get()
        p.join()
        return result

    return process


def run_with_tf_optimizations(do_eager_mode, do_xla):
    def run_func(func):
        @wraps(func)
        def run_in_eager_mode(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        @tf.function(experimental_compile=do_xla)
        def run_in_graph_mode(*args, **kwargs):
            return func(*args, **kwargs)

        if do_eager_mode is True:
            assert (
                do_xla is False
            ), "Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`."
            return run_in_eager_mode
        else:
            return run_in_graph_mode

    return run_func


def random_input_ids(batch_size, sequence_length, vocab_size):
    rng = random.Random()
    values = [rng.randint(0, vocab_size - 1) for i in range(batch_size * sequence_length)]
    return tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)


class TensorflowBenchmark(Benchmark):

    args: TensorflowBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "Tensorflow"

    @property
    def framework_version(self):
        return tf.__version__

    @run_on_separate_process
    def inference_speed(self, model_name, batch_size, sequence_length):
        # initialize GPU on separate process
        strategy = self.args.strategy
        assert strategy is not None, "A device strategy has to be initialized before using Tensorflow."
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_inference)

    def train_speed(self, model_name, batch_size, sequence_length):
        raise NotImplementedError(
            "Training is currently not really implemented." "Wait for TFTrainer to support CLM and MLM."
        )

    @run_on_separate_process
    def inference_memory(self, model_name, batch_size, sequence_length):
        # initialize GPU on separate process
        if self.args.is_gpu:
            tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
        strategy = self.args.strategy
        assert strategy is not None, "A device strategy has to be initialized before using Tensorflow."
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_inference)

    def train_memory(self, model_name, batch_size, sequence_length):
        raise NotImplementedError(
            "Training is currently not really implemented. Wait for TFTrainer to support CLM and MLM."
        )

    def _prepare_inference_func(self, model_name, batch_size, sequence_length):
        config = self.config_dict[model_name]

        if self.args.with_lm_head:
            model = TF_MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)
        else:
            model = TF_MODEL_MAPPING[config.__class__](config)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_decoder_forward():
            model(input_ids, decoder_input_ids=input_ids, training=False)

        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_forward():
            model(input_ids, training=False)

        _inference = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward

        return _inference

    def _measure_speed(self, func):
        with self.args.strategy.scope():
            try:
                if self.args.is_tpu or self.args.use_xla:
                    # run additional 10 times to stabilize compilation for tpu
                    logger.info("Do inference on TPU. Running model 5 times to stabilize compilation")
                    timeit.repeat(func, repeat=1, number=5)

                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(func, repeat=self.args.repeat, number=10,)

                return min(runtimes) / 10.0
            except ResourceExhaustedError as e:
                self.print_fn("Doesn't fit on GPU. {}".format(e))

    def _measure_memory(self, func):
        logger.info(
            "Note that Tensorflow allocates more memory than"
            "it might need to speed up computation."
            "The memory reported here corresponds to the memory"
            "reported by `nvidia-smi`, which can vary depending"
            "on total available memory on the GPU that is used."
        )
        with self.args.strategy.scope():
            try:
                if self.args.trace_memory_line_by_line:
                    assert (
                        self.args.eager_mode
                    ), "`args.eager_mode` is set to `False`. Make sure to run model in eager mode to measure memory consumption line by line."
                    trace = start_memory_tracing("transformers")

                if not self.args.no_tpu and self.args.is_tpu:
                    # tpu
                    raise NotImplementedError(
                        "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.no_memory=True`"
                    )
                if not self.args.is_gpu:
                    # cpu
                    if self.args.trace_memory_line_by_line:
                        logger.info(
                            "When enabling line by line tracing, the max peak memory for CPU is inaccurate in Tensorflow."
                        )
                        memory = None
                    else:
                        memory_bytes = measure_peak_memory_cpu(func)
                        memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes
                memory = 0
                if self.args.is_gpu:
                    # gpu
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

                if self.args.trace_memory_line_by_line:
                    summary = stop_memory_tracing(trace)
                    if memory is None:
                        memory.summary.total_memory
                else:
                    summary = None

                return memory, summary
            except ResourceExhaustedError as e:
                self.print_fn("Doesn't fit on GPU. {}".format(e))
                return "N/A", None
