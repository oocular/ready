"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""  # noqa: E501

import os

# import random
import cupy as cp
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.gxf import Entity
from holoscan.operators import (FormatConverterOp, HolovizOp, InferenceOp,
                                SegmentationPostprocessorOp,
                                V4L2VideoCaptureOp, VideoStreamReplayerOp)
from holoscan.resources import (BlockMemoryPool, CudaStreamPool,
                                MemoryStorageType, UnboundedAllocator)


class InfoOp(Operator):
    """
    Information Operator

    Input:
        in: source_video

    Output:
        out: 'dynamic_text': array([[0.01, 0.01, 0.05]])
        output_specs: holoscan.operators.holoviz._holoviz.HolovizOp.InputSpec object
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
        print(f"in_message={in_message}")
        tensor = cp.asarray(in_message.get("source_video"))
        print(f"tensor.shape={tensor.shape}")
        print(f"tensor.min()={cp.min(tensor)}")
        print(f"tensor.max()={cp.max(tensor)}")
        print(f"tensor.mean()={cp.mean(tensor)}")
        tensor_without_alpha = tensor[:, :, :3]
        print(f"tensor_without_alpha.shape={tensor_without_alpha.shape}")
        print(f"tensor_without_alpha.min()={cp.min(tensor_without_alpha)}")
        print(f"tensor_without_alpha.max()={cp.max(tensor_without_alpha)}")
        print(f"tensor_without_alpha.mean()={cp.mean(tensor_without_alpha)}")

        dynamic_text = cp.asarray(
            [
                (0.01, 0.01, 0.05),  # (x, y, font_size)
            ],
        )
        out_message = {
            "dynamic_text": dynamic_text,
        }
        print(f"out_message: {out_message}")
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
            + " tensor.min()="
            + str(cp.min(tensor))
            + " tensor.max()="
            + str(cp.max(tensor))
            + " tensor.mean()="
            + str(cp.mean(tensor))
        ]
        spec.color = [
            1,
            1,
            1,
            # random.uniform(0.0, 1.0),
            0.99,
        ]

        spec.views = [view]
        specs.append(spec)
        print(f"specs: {specs}")
        op_output.emit(specs, "output_specs")

        self.frame_count += 1


# Now define a simple application using the operators defined above
class App(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - V4L2VideoCaptureOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the ImageProcessingOp.
    The HolovizOp displays the processed frames.
    """

    def compose(self):
        source_args = self.kwargs("source")
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        if "width" in source_args and "height" in source_args:
            # width and height given, use BlockMemoryPool (better latency)
            width = source_args["width"]
            height = source_args["height"]
            #####REMVOEprint(f'xxxxxxxxxxxxxxxxx {width} {height}')
            n_channels = 4  # RGBA
            bpp = 4  # bytes per pixel
            block_size = width * height * n_channels
            drop_alpha_block_size = width * height * n_channels * bpp
            drop_alpha_num_blocks = 2
            # REMOVE allocator = BlockMemoryPool(
            # REMOVE    self, name="pool", storage_type=0, block_size=block_size, num_blocks=1
            # REMOVE )
            allocator = BlockMemoryPool(
                self,
                name="pool",
                storage_type=0,  # storage_type=MemoryStorageType.DEVICE,
                block_size=drop_alpha_block_size,
                num_blocks=drop_alpha_num_blocks,
            )

            source = V4L2VideoCaptureOp(
                self,
                name="source",
                allocator=allocator,
                **source_args,
            )

            # Set Holoviz width and height from source resolution
            visualizer_args = self.kwargs("visualizer")
            visualizer_args["width"] = width
            visualizer_args["height"] = height
            visualizer = HolovizOp(
                self,
                name="visualizer",
                # ?allocator=allocator,
                cuda_stream_pool=cuda_stream_pool,
                **visualizer_args,
            )
        else:
            # width and height not given, use UnboundedAllocator (worse latency)
            source = V4L2VideoCaptureOp(
                self,
                name="source",
                allocator=UnboundedAllocator(self, name="pool"),
                **self.kwargs("source"),
            )
            visualizer = HolovizOp(
                self,
                name="visualizer",
                **self.kwargs("visualizer"),
            )

        info_op = InfoOp(
            self,
            name="info_op",
            allocator=host_allocator,
            **self.kwargs("info_op"),
        )

        drop_alpha_channel = FormatConverterOp(
            self,
            name="drop_alpha_channel",
            pool=host_allocator,
            # out_dtype=,
            # in_dtype=,
            # in_tensor_name=,
            # out_tensor_name=,
            # scale_min=,
            # scale_max=,
            # alpha_value=,
            # resize_height=,
            # resize_width=,
            # resize_mode=,
            # out_channel_order=,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("drop_alpha_channel"),
        )

        # Define the workflow
        self.add_flow(source, visualizer, {("signal", "receivers")})
        self.add_flow(source, drop_alpha_channel, {("signal", "source_video")})
        self.add_flow(drop_alpha_channel, info_op, {("", "in")})
        self.add_flow(info_op, visualizer, {("out", "receivers")})
        self.add_flow(info_op, visualizer, {("output_specs", "input_specs")})


def main(config_file):
    app = App()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()
    print("Application has finished running.")


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "v4l2_camera.yaml")
    main(config_file=config_file)
