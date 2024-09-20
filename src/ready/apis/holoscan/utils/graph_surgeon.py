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

import onnx
import onnx_graphsurgeon as gs


def main():
    """
    Rebinding model to new nodes (NCHW to NHWC)

    Input model: sys.argv[1]
    Output model: sys.argv[2]
    Channels: sys.argv[3]
    High: sys.argv[4]
    Width: sys.argv[5]
    :return:

    https://github.com/nvidia-holoscan/holohub/blob/main/applications/ssd_detection_endoscopy_tools/scripts/graph_surgeon_ssd.py
    """
    # TODO use only path and input model, instead of input model and output model
    graph = gs.import_onnx(onnx.load(sys.argv[1]))

    # Update graph input/output names
    graph.inputs[0].name += "_old"
    graph.outputs[0].name += "_old"

    channel = int(sys.argv[3])
    height = int(sys.argv[4])
    width = int(sys.argv[5])

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

    onnx.save(gs.export_onnx(graph), sys.argv[2])


if __name__ == "__main__":
    main()
