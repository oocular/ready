# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from argparse import ArgumentParser

import onnx
import onnx_graphsurgeon as gs

if __name__ == "__main__":
    """
    Script to rebind model to new nodes (NCHW to NHWC)

    Usage:
        Run this script from the root directory of the project:
        python  -p <MODEL_PATH> -m <model_name.pth> -c <channels> -h <height> -w <width>

    Arguments:
        -p, --model_path: Set the model path. Default is none.
        -m, --input_model_name: Set input model name. Default is none.
        -c, --channel: Set the number of channels. Default is none.
        -h, --height: Set the height. Default is none.
        -w, --width: Set the width. Default is none.

    Reference:
    https://github.com/nvidia-holoscan/holohub/blob/main/applications/ssd_detection_endoscopy_tools/scripts/graph_surgeon_ssd.py
    """
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument("-p",
                        "--model_path",
                        type=str,
                        help="Set the model path.")
    parser.add_argument("-m",
                        "--input_model_name",
                        type=str,
                        help="Set input model name.")
    parser.add_argument("-ch",
                        "--channel",
                        type=int,
                        help="Set the number of channels.")
    parser.add_argument("-he",
                        "--height",
                        type=int,
                        help="Set the height.")
    parser.add_argument("-wi",
                        "--width",
                        type=int,
                        help="Set the width.")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    INPUT_MODEL_NAME = args.input_model_name
    channel = args.channel
    height = args.height
    width = args.width
    MODEL_NAME = INPUT_MODEL_NAME[:-4]

    graph = gs.import_onnx(onnx.load(MODEL_PATH+"/"+ MODEL_NAME+"-sim.onnx"))

    # Update graph input/output names
    graph.inputs[0].name += "_old"
    graph.outputs[0].name += "_old"


    # Insert a transpose at the network input tensor [1, 3, width, height] and rebind it to the
    # new node [1, height, width, 3] be careful which one is h and which one is w
    nhwc_to_nchw_in = gs.Node(
        "Transpose", name="transpose_input", attrs={"perm": [0, 3, 1, 2]}
    )
    nhwc_to_nchw_in.outputs = graph.inputs
    graph.inputs = [
        gs.Variable(
            "INPUT__0", dtype=graph.inputs[0].dtype, shape=[1, height, width, channel]
        )
    ]
    nhwc_to_nchw_in.inputs = graph.inputs

    graph.nodes.extend([nhwc_to_nchw_in])
    graph.toposort().cleanup()

    onnx.save(gs.export_onnx(graph), MODEL_PATH+"/"+ MODEL_NAME+"-sim-BHWC.onnx")
