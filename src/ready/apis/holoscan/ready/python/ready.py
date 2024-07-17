# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cupy as cp
from argparse import ArgumentParser

#from holoscan.core import Application, Tracker
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.gxf import Entity
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
#from holoscan.resources import UnboundedAllocator
from holoscan.resources import BlockMemoryPool, CudaStreamPool, UnboundedAllocator

class InfoOp(Operator):
    """
    Information Operator

    Input:

    Output:
    """

    def __init__(self, *args, **kwargs):
        """Initialize Operator"""
        self.frame_count = 1
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("in")
        spec.output("out")
        spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        print(f"---------- InfoOp  ------------")
        in_message = op_input.receive("in")
        #print(f"in_message={in_message}")
        tensor = cp.asarray(in_message.get("output"), dtype=cp.float32)
        print(f"tensor.shape={tensor.shape}")

        out_message = Entity(context)
	##out_message?
        #op_output.emit(out_message, "out")

        dynamic_text = cp.asarray(
            [
                (0.01, 0.01, 0.05),  # (x, y, font_size)
            ],
        )
        out_message = {
            "dynamic_text": dynamic_text,
        }
        #print(f"out_message: {out_message}")
        op_output.emit(out_message, "out")

        specs = []
        spec = HolovizOp.InputSpec("dynamic_text", HolovizOp.InputType.TEXT)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.width = 1.25
        view.height = 1.25

        spec.text = [
            "Frame "
            + str(self.frame_count)
            #+ " tensor.min()="
            #+ str(cp.min(tensor))
            #+ " tensor.max()="
            #+ str(cp.max(tensor))
            #+ " tensor.mean()="
            #+ str(cp.mean(tensor))
        ]
        spec.color = [
            1,
            1,
            1,
            #random.uniform(0.0, 1.0),
            0.99,
        ]

        spec.views = [view]
        specs.append(spec)
        #print(f"specs: {specs}")
        op_output.emit(specs, "output_specs")

        self.frame_count += 1


class READYApp(Application):
    def __init__(self, data=None, model_name=None):
        """Initialize the application

        Parameters
        ----------
        data : Location to the data
        model_name : Model name
        """

        super().__init__()

        self.name = "READY App"
        self.data_path = data

        if data == "none":
            data = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")
        else:
            self.video_dir = os.path.join(self.data_path, "videos")
            if not os.path.exists(self.video_dir):
                raise ValueError(f"Could not find video data: {self.video_dir=}")
            self.models_path = os.path.join(self.data_path, "models")
            if not os.path.exists(self.models_path):
                raise ValueError(f"Could not find models data: {self.models_path=}")

        self.model_name = model_name
        self.models_path_map = {
            "ready_model": os.path.join(self.models_path, self.model_name),
        }


    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        source = VideoStreamReplayerOp(
            self, name="replayer", directory=self.video_dir, **self.kwargs("replayer")
        )

        preprocessor = FormatConverterOp(
            self, name="preprocessor", 
            pool=host_allocator, 
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("preprocessor")
        )

        info_op = InfoOp(
            self,
            name="info_op",
            allocator=host_allocator,
            **self.kwargs("info_op"),
        )

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=self.models_path_map,
            **self.kwargs("inference"),
        )

        segpostprocessor = SegmentationPostprocessorOp(
            self, name="segpostprocessor", allocator=host_allocator, **self.kwargs("segpostprocessor")
        )

        viz = HolovizOp(self, name="viz", cuda_stream_pool=cuda_stream_pool, **self.kwargs("viz"))

        # Workflow
        self.add_flow(source, viz, {("output", "receivers")})
        self.add_flow(source, preprocessor, {("output", "source_video")})
        self.add_flow(preprocessor, inference, {("tensor", "receivers")})
        self.add_flow(inference, segpostprocessor, {("transmitter", "")}) #OR {("transmitter", "in_tensor")})
        self.add_flow(inference, info_op, {("", "")})
        self.add_flow(segpostprocessor, viz, {("out_tensor", "receivers")})
        self.add_flow(info_op, viz, {("out", "receivers")})
        self.add_flow(info_op, viz, {("output_specs", "input_specs")})

if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default="none",
        help=("Set model name"),
    )
    parser.add_argument(
        "-l",
        "--logger_filename",
        default="logger.log",
        help=("Set logger filename"),
    )
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "ready.yaml")

    app = READYApp(data=args.data, model_name=args.model_name)
    with Tracker(app, filename=args.logger_filename) as tracker:
       app.config(config_file)
       app.run()

