# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Holoscan BYOM application"""

import os
from argparse import ArgumentParser

from holoscan.core import Application, Tracker
from holoscan.operators import (FormatConverterOp, HolovizOp, InferenceOp,
                                SegmentationPostprocessorOp,
                                VideoStreamReplayerOp)
from holoscan.resources import UnboundedAllocator


class BYOMApp(Application):
    """BYOM Application"""
    def __init__(self, data=None, model_name=None):
        """Initialize the application

        Parameters
        ----------
        data : Location to the data
        model_name : Model name
        """

        super().__init__()

        self.name = "BYOM App"
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
            "byom_model": os.path.join(self.models_path, self.model_name),
        }

    def compose(self):
        """Compose the application"""
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        source = VideoStreamReplayerOp(
            self, name="replayer", directory=self.video_dir, **self.kwargs("replayer")
        )

        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=host_allocator,
            **self.kwargs("preprocessor"),
        )

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=self.models_path_map,
            **self.kwargs("inference"),
        )

        postprocessor = SegmentationPostprocessorOp(
            self,
            name="postprocessor",
            allocator=host_allocator,
            **self.kwargs("postprocessor"),
        )

        viz = HolovizOp(self, name="viz", **self.kwargs("viz"))

        # Workflow
        self.add_flow(source, viz, {("output", "receivers")})
        self.add_flow(source, preprocessor, {("output", "source_video")})
        self.add_flow(preprocessor, inference, {("tensor", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in_tensor")})
        self.add_flow(postprocessor, viz, {("out_tensor", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="BYOM demo application.")
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

    config_file = os.path.join(os.path.dirname(__file__), "byom.yaml")

    app = BYOMApp(data=args.data, model_name=args.model_name)
    with Tracker(app, filename=args.logger_filename) as tracker:
        app.config(config_file)
        app.run()
