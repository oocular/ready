
"""Holoscan READY application"""

import os
from argparse import ArgumentParser

import cupy as cp
import cv2
import holoscan as hs
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.gxf import Entity
from holoscan.operators import (FormatConverterOp, HolovizOp, InferenceOp,
                                SegmentationPostprocessorOp,
                                V4L2VideoCaptureOp, VideoStreamReplayerOp)
from holoscan.resources import (BlockMemoryPool, CudaStreamPool,
                                UnboundedAllocator, MemoryStorageType)


class PreInfoOp(Operator):
    """
    Pre Information Operator

    Input:

    Output:
    """

    def __init__(self, *args, **kwargs):
        """Initialize Operator"""
        # self.frame_count = 1
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("source_video")
        spec.output("out")
        # spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        print(f" \/ \/ \/ \/ \/ \/ ")
        print(f"   PreInfoOp  ")
        in_message = op_input.receive("source_video")
        tensor = cp.asarray(in_message.get(""), dtype=cp.float32)
        tensor_1ch = tensor[:, :, 0]
        print(f"tensor.shape={tensor.shape}")
        print(f"tensor_1ch.shape={tensor_1ch.shape}")

        out_message = Entity(context)
        out_message.add(hs.as_tensor(tensor_1ch), "tensor1ch")
        # out_message.add(hs.as_tensor(tensor), "tensor")
        op_output.emit(in_message, "out")


class FormatInferenceInputOp(Operator):
    """
    FormatInferenceInputOp

    Input:
        tensor.shape=(400, 640, 3)
    Output:
        tensor_.shape=(1, 1, 400, 640)
    """

    def __init__(self, *args, **kwargs):
        """Initialize Operator"""
        # self.frame_count = 1
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        print(f" \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ ")
        print(f"   FormatInferenceInputOp  ")
        in_message = op_input.receive("in")
        # print(in_message)
        tensor = cp.asarray(in_message.get("out_preprocessor"), dtype=cp.float32)
        print(f"**")
        print(f"tensor.shape={tensor.shape}")  # tensor.shape=(400, 640, 3)

        print(f"tensor.min {cp.min(tensor)}")
        print(f"tensor.max {cp.max(tensor)}")
        print(f"tensor.mean {cp.mean(tensor)}")

        tensor_ = cp.moveaxis(tensor, 2, 0)[None]
        print(f"**")
        print(f"tensor_.shape={tensor_.shape}")  # tensor_.shape=(1, 3, 400, 640)
        print(f"tensor_.min {cp.min(tensor_)}")
        print(f"tensor_.max {cp.max(tensor_)}")
        print(f"tensor_.mean {cp.mean(tensor_)}")

        tensor_1ch = tensor_[:, 0, :, :]
        print(f"**")
        print(f"tensor_1ch.shape={tensor_1ch.shape}")  # tensor_1ch.shape=(1, 400, 640)
        print(f"tensor1ch.min {cp.min(tensor_1ch)}")
        print(f"tensor1ch.max {cp.max(tensor_1ch)}")
        print(f"tensor1ch.mean {cp.mean(tensor_1ch)}")

        tensor_1cchh = cp.expand_dims(tensor_1ch, 0)
        print(f"**")
        print(
            f"tensor_1CH.shape={tensor_1cchh.shape}"
        )  # tensor_1CH.shape=(1, 1, 400, 640)
        print(f"tensor_1CH.min {cp.min(tensor_1cchh)}")
        print(f"tensor_1CH.max {cp.max(tensor_1cchh)}")
        print(f"tensor_1CH.mean {cp.mean(tensor_1cchh)}")

        tensor_1ccchhh = cp.expand_dims(tensor_1ch, -1)
        print(f"**")
        print(
            f"tensor_1Ch.shape={tensor_1ccchhh.shape}"
        )  # tensor_1CH.shape=(1, 400, 640, 1)
        print(f"tensor_1Ch.min {cp.min(tensor_1ccchhh)}")
        print(f"tensor_1Ch.max {cp.max(tensor_1ccchhh)}")
        print(f"tensor_1Ch.mean {cp.mean(tensor_1ccchhh)}")

        out_message = Entity(context)
        out_message.add(hs.as_tensor(tensor_), "out_preprocessor")
        # out_message.add(hs.as_tensor(tensor_1Ch), "out_preprocessor")
        op_output.emit(out_message, "out")


def cv_cuda_gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    """ cv cuda gpumat from cp array """
    print("::: cv_cuda_gpumat_from_cp_array :::")
    print(arr.__cuda_array_interface__["shape"])
    print(arr.__cuda_array_interface__["data"][0])
    type_map = {
        cp.dtype("uint8"): cv2.CV_8U,
        cp.dtype("int8"): cv2.CV_8S,
        cp.dtype("uint16"): cv2.CV_16U,
        cp.dtype("int16"): cv2.CV_16S,
        cp.dtype("int32"): cv2.CV_32S,
        cp.dtype("float32"): cv2.CV_32F,
        cp.dtype("float64"): cv2.CV_64F,
    }
    print(f"arr.dtype: {arr.dtype}")  # float32
    depth = type_map.get(arr.dtype)
    assert depth is not None, "Unsupported CuPy array dtype"
    print(f" depth: {depth}")
    channels = 1 if len(arr.shape) == 2 else arr.shape[1]
    mat_type = depth + ((channels - 1) << 3)
    print(mat_type)

    mat = cv2.cuda.createGpuMatFromCudaMemory(
        arr.__cuda_array_interface__["shape"],
        mat_type,
        arr.__cuda_array_interface__["data"][0],
    )
    # print(type(mat)) #<class 'cv2.cuda.GpuMat'>
    print(mat.size())  # (400, 640)
    print(mat.channels())  # 1
    print("::: cv_cuda_gpumat_from_cp_array :::")
    return mat


class PostInferenceOp(Operator):
    """
    Post Inference Operator

    Input:

    Output:
    """

    # def __init__(self, *args, **kwargs):
    def __init__(self, fragment, width=None, height=None, **kwargs):
        """Initialize Operator"""
        super().__init__(fragment, **kwargs)
        self.frame_count = 1
        self.width = width
        self.height = height
        self.x_centroid_coords = cp.array([0])
        self.y_centroid_coords = cp.array([0])
        self.cycle = 0
        # Arrays for 30 frames
        self.x = cp.linspace(0, 1.0, 30)
        self.y = cp.linspace(0, 1.0, 30)
        self.x_varing_array = cp.zeros((30, 2))
        self.y_varing_array = cp.zeros((30, 2))

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("in")
        spec.output("out")
        spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        print(f" \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ ")
        print(f"   PostInferenceOperator  ")
        in_message = op_input.receive("in")
        # print(f"in_message={in_message}")
        tensor = cp.asarray(in_message.get("unet_out"), dtype=cp.float32)
        print(f"unet_out tensor.shape={tensor.shape}")  # tensor.shape=(1, 4, 400, 640)

        tensor_1ch_background = tensor[:, 0, :, :]
        tensor_1ch_sclera = tensor[:, 1, :, :]
        tensor_1ch_iris = tensor[:, 2, :, :]

        print(f"shape of tensor_1ch_background: {tensor_1ch_background.shape}")
        print(f"shape of tensor_1ch_sclera: {tensor_1ch_sclera.shape}")
        print(f"shape of tensor_1ch_iris: {tensor_1ch_iris.shape}")

        ### CENTROID OF PUPIL MASK
        tensor_1ch_pupil = tensor[:, 3, :, :]
        print(f"tensor.min {cp.min(tensor_1ch_pupil)}")
        print(f"tensor.max {cp.max(tensor_1ch_pupil)}")
        print(f"tensor.mean {cp.mean(tensor_1ch_pupil)}")

        print(
            f"tensor_1ch_pupil.shape={tensor_1ch_pupil.shape}"
        )  # tensor.shape=(1, 400, 640)
        tensor_1ch_pupil_sq = cp.squeeze(tensor_1ch_pupil, axis=None)
        print(
            f"tensor_1ch_pupil_sq.shape={tensor_1ch_pupil_sq.shape}"
        )  # tensor.shape=(400, 640)
        tensor_1ch_pupil_sq_uint8 = tensor_1ch_pupil_sq.astype(cp.uint8)
        print(tensor_1ch_pupil_sq_uint8.dtype)  # uint8

        mask_pupil_bool = tensor_1ch_pupil_sq_uint8 > 1
        print(
            f"mask_pupil_bool.shape {mask_pupil_bool.shape}"
        )  # tensor.shape=(1, 4, 400, 640)
        print(f"mask_pupil_bool.dtype {mask_pupil_bool.dtype}")  # bool

        # /usr/local/lib/python3.10/dist-packages/numpy/core/getlimits.py:500:
        # UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
        #  setattr(self, word, getattr(machar, word).flat[0])
        # /usr/local/lib/python3.10/dist-packages/numpy/core/getlimits.py:89:
        # UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
        #  return self._float_to_str(self.smallest_subnormal)

        centroid = cp.mean(cp.argwhere(mask_pupil_bool), axis=0)
        centroid = cp.nan_to_num(centroid)  # convert float NaN to integer
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
        print(f"centroid: {centroid}")
        # https://stackoverflow.com/questions/73131778/

        # 	#EXPERIMENTAL (to be removed or checked for multiple mask)
        #        #https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
        #        frameUMat = cv2.UMat(tensor_1ch_pupil_sq_uint8.shape[0],
        #               tensor_1ch_pupil_sq_uint8.shape[1], cv2.CV_8U)
        #        mask_gray = cv2.normalize(src=frameUMat, dst=None, alpha=0, beta=255,
        #               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #        #print(mask_gray.get())
        #        blur = cv2.GaussianBlur(mask_gray, (5, 5), 0)
        #        ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
        #        # calculate x,y coordinate of center
        #        M = cv2.moments(thresh)
        #        #M = cv2.cuda.moments(thresh)
        #        #multiple objects
        #        #contours, hierarchies =
        #               cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #        #M = cv2.moments(contours)
        #        print(f"Moments from threshold")
        #        print(M["m00"])
        #        print(M["m01"])
        #        print(M["m10"])
        #        if M["m00"]!=0:
        #           cX = int(M["m10"] / M["m00"])
        #           cY = int(M["m01"] / M["m00"])
        #           print(f'cX: {cX} cY: {cY}')
        #           print(f'cX: {cX/400} cY: {cY/640}')
        #
        #        #USING GPUMat with opencv
        #        #cv_gpumat = cv_cuda_gpumat_from_cp_array(tensor_1ch_pupil_sq_uint8)
        #        #print(type(cv_gpumat)) #<class 'cv2.cuda.GpuMat'>

        centroid_xy = cp.asarray(
            [
                (centroid_x / self.width, centroid_y / self.height),
            ],
            dtype=cp.float32,
        )
        # print(centroid_xy) #[[320. 200.]]
        centroid_xy = centroid_xy[cp.newaxis, :, :]
        print(f"normalised centroid_xy: {centroid_xy}")

        out_message = Entity(context)

        text_coords = cp.asarray(
            [
                # (0.01, 0.01, 0.03),  # (x, y, font_size)
                (centroid_x / self.width, centroid_y / self.height, 0.03),
            ],
            dtype=cp.float32,
        )
        text_coords = text_coords[cp.newaxis, :, :]

        #######################################
        # Create a time-varying tensors for "centroid coordinate points"
        #
        # Set of (x, y) points with 30 points equally spaced along x
        # whose y coordinate varies based on
        # self.y_centroid_coords, and
        # self.x_centroid_coords
        # over time.
        #
        self.y_centroid_coords = cp.append(self.y_centroid_coords, centroid_y / self.height)
        self.x_centroid_coords = cp.append(self.x_centroid_coords, centroid_x / self.width)
        # print(self.y_centroid_coords)
        if self.frame_count % 29 == 0:
            #print(f"self.y_centroid_coords: {self.y_centroid_coords}")
            self.x_varing_array = cp.stack((self.x, self.y_centroid_coords), axis=-1)
            self.y_varing_array = cp.stack((self.x, self.x_centroid_coords), axis=-1)
            #print(f"self.x_varing_array: {self.x_varing_array}")
            self.cycle = 0
            self.y_centroid_coords = cp.array([0]) #clean array
            self.x_centroid_coords = cp.array([0]) #clean array

        self.cycle += 1

        # adds messages
        out_message.add(hs.as_tensor(centroid_xy), "pupil_cXcY")
        out_message.add(hs.as_tensor(text_coords), "text_coords")
        out_message.add(hs.as_tensor(self.x_varing_array), "x_coords_varing_array")
        out_message.add(hs.as_tensor(self.y_varing_array), "y_coords_varing_array")
        op_output.emit(out_message, "out")

        specs = []
        spec = HolovizOp.InputSpec("text_coords", HolovizOp.InputType.TEXT)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.02
        view.offset_y = 0.02
        view.width = 1.0
        view.height = 1.0

        spec.text = [
            "Frame "
            + str(self.frame_count)
            + "\n"
            + " unetout shape="
            + str(tensor.shape)
            + "\n"
            + " t.min="
            + str(cp.min(tensor))
            + " t.max="
            + str(cp.max(tensor))
            + " t.mean="
            + str(cp.mean(tensor))
            + "\n"
            + " cx="
            + str(centroid_x / self.width)
            + " cy="
            + str(centroid_y / self.height)
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
        op_output.emit(specs, "output_specs")

        self.frame_count += 1


class READYApp(Application):
    def __init__(
        self, source=None, data=None, model_name=None, debug_print_flag=None
    ):
        """Initialize the application

        Parameters
        ----------
        data : Location to the data
        model_name : Model name
        """

        # super().__init__(**kwargs)
        super().__init__()

        self.name = "READY App"
        self.source = source
        self.debug_print_flag = debug_print_flag
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
        source_args = self.kwargs("source")

        if self.source.lower() == "replayer":
            width = 640  # TODO source_args["width"]
            height = 400  # TODO source_args["height"]
            n_channels = 1
            bpp = 1  # bytes per pixel
            block_size = width * height * n_channels * bpp
            num_blocks = 1
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                ## [error] [block_memory_pool.cpp:125] Requested 768000 bytes of memory in a pool with block size 512000
                # allocator=BlockMemoryPool(
                #     self,
                #     name="video_replayer_pool",
                #     storage_type=0,
                #     # storage_type=MemoryStorageType.DEVICE,
                #     block_size=block_size,
                #     num_blocks=num_blocks,
                # ),
                allocator=host_allocator,
                basename= "video_3framesx10",
                directory=self.video_dir,
                frame_rate=0.0,
                realtime=True, # default: true
                repeat=True, # default: false
                count=0, # default: 0 (no frame count restriction)
                # **self.kwargs("replayer"),
            )

        elif self.source.lower() == "v4l2":
            width = 640  # TODO source_args["width"]
            height = 400  # TODO source_args["height"]
            n_channels = 4  # RGBA
            bpp = 4  # bytes per pixel
            drop_alpha_block_size = width * height * n_channels * bpp
            drop_alpha_num_blocks = 2
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                # allocator=v4l2_allocator,
                allocator=BlockMemoryPool(
                    self,
                    name="v4l2_replayer_pool",
                    storage_type=0,
                    # storage_type=MemoryStorageType.DEVICE, #RuntimeError: Failed to allocate output buffer.
                    block_size=drop_alpha_block_size,
                    num_blocks=drop_alpha_num_blocks,
                ),
                device="/dev/video0",
                width=640,
                height=480,
                pixel_format="YUYV",
                # **self.kwargs("v4l2_source"),
            )

        else:
            print(f"plesea either choose v4l2 or replayer")

        formatter_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        pre_info_op_replayer = PreInfoOp(
            self,
            name="pre_info_op_replayer",
            allocator=host_allocator,
            **self.kwargs("pre_info_op_replayer"),
        )

        in_dtype = "rgb888" # float32
        bytes_per_float32 =4
        in_components=3
        preprocessor_replayer = FormatConverterOp(
            self,
            name="preprocessor_replayer",
            in_dtype=in_dtype,
            # pool=UnboundedAllocator(self, name="FormatConverter allocator"),
            pool=BlockMemoryPool(
                self,
                name="preprocessor_replayer_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=640 * 400 * bytes_per_float32 * in_components,
                num_blocks=2*3,
            ),
            out_tensor_name="out_preprocessor",
            scale_min=1.0,
            scale_max=252.0,
            resize_width=640,
            resize_height=400,
            out_dtype="float32",
            cuda_stream_pool=formatter_cuda_stream_pool,
            # **self.kwargs("preprocessor_replayer"),
        )

        preprocessor_v4l2 = FormatConverterOp(
            self,
            name="preprocessor_v4l2",
            out_tensor_name="out_preprocessor",            
            in_dtype="rgba8888", #for four channels
            out_dtype="float32",
            scale_min=1.0,
            scale_max=252.0,
            resize_width=640,
            resize_height=400,            
            # pool=host_allocator,
            pool=BlockMemoryPool(
                self,
                name="preprocessor_replayer_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=640 * 400 * bytes_per_float32 * in_components,
                num_blocks=2*3,
            ),
            cuda_stream_pool=formatter_cuda_stream_pool,
            **self.kwargs("preprocessor_v4l2"),
        )

        format_input = FormatInferenceInputOp(
            self,
            name="format_input",
            allocator=host_allocator,
            **self.kwargs("format_input"),
        )


        n_channels_inference = 4
        width_inference = 640
        height_inference = 400
        bpp_inference = 4
        inference_block_size = (
            width_inference * height_inference * n_channels_inference * bpp_inference
        )
        inference_num_blocks = 2

        # inference_allocator=UnboundedAllocator(self, name="inference_allocator"),
        # the RMMAllocator supported since v2.6 is much faster than the default UnboundAllocator
        try:
            from holoscan.resources import RMMAllocator
            allocator = RMMAllocator(self, name="inference_allocator")
        except Exception:
            pass

        inference = InferenceOp(
            self,
            name="inference",
            backend="trt",
            pre_processor_map={"ready_model": ["out_preprocessor"]},
            inference_map={"ready_model": ["unet_out"]},
            enable_fp16=False,
            parallel_inference=True,
            infer_on_cpu=False,
            input_on_cuda=True,
            output_on_cuda=True,
            transmit_on_cuda=True,
            is_engine_path=False,
            # allocator=allocator,
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=inference_block_size,
                num_blocks=inference_num_blocks,
            ),
            model_path_map=self.models_path_map,
            # **self.kwargs("inference"),
        )

        post_inference_op = PostInferenceOp(
            self,
            name="post_inference_op",
            allocator=host_allocator,
            width=width,
            height=height,
            **self.kwargs("post_inference_op"),
        )

        segpostprocessor = SegmentationPostprocessorOp(
            self,
            name="segpostprocessor",
            allocator=host_allocator,
            in_tensor_name="unet_out",
            network_output_type="softmax",
            data_format="nchw",
            # **self.kwargs("segpostprocessor"),
        )

        viz = HolovizOp(
            self,
            name="viz",
            window_title="READY demo",
            width=640,
            height=400,
            tensors=[
                dict(
                    name="",
                    type="color",
                    opacity=1.0,
                    priority= 0,
                ),
                dict(
                    name="pupil_cXcY",
                    type="crosses",
                    color=[0.6, 0.1, 0.6, 0.8],
                    opacity=0.85,
                    priority=2,
                    line_width=5.0, #for crosses only
                ),
                dict(
                    name="x_coords_varing_array",
                    type="points",
                    color=[1.0, 0.0, 0.0, 1.0],
                    point_size=5.0,
                    priority=2,
                ),
                dict(
                    name="y_coords_varing_array",
                    type="points",
                    color=[0.0, 1.0, 0.0, 1.0],
                    point_size=5.0,
                    priority=2,
                ),
                dict(
                    name="out_tensor",
                    type="color_lut",
                    opacity=1.0,
                    priority=0,
                ),
            ],
            color_lut=[[0.65, 0.81, 0.89, 0.01],[0.3, 0.3, 0.9, 0.5],[0.1, 0.8, 0.2, 0.5],[0.9, 0.9, 0.3, 0.8],]
            # **self.kwargs("viz"),
        )

        if self.source.lower() == "replayer":
            self.add_flow(source, viz, {("", "receivers")})

            self.add_flow(source, pre_info_op_replayer, {("output", "source_video")})
            self.add_flow(pre_info_op_replayer, preprocessor_replayer, {("", "")})

            self.add_flow(preprocessor_replayer, format_input)
            self.add_flow(format_input, inference, {("", "receivers")})

            self.add_flow(inference, segpostprocessor, {("transmitter", "")})
            self.add_flow(segpostprocessor, viz, {("", "receivers")})

            self.add_flow(inference, post_inference_op, {("", "in")})
            self.add_flow(post_inference_op, viz, {("out", "receivers")})
            self.add_flow(post_inference_op, viz, {("output_specs", "input_specs")})

        elif self.source.lower() == "v4l2":
            self.add_flow(source, viz, {("signal", "receivers")})

            self.add_flow(source, preprocessor_v4l2, {("signal", "source_video")})
            self.add_flow(preprocessor_v4l2, inference, {("tensor", "receivers")})

            self.add_flow(inference, segpostprocessor, {("transmitter", "")})
            self.add_flow(segpostprocessor, viz, {("", "receivers")})

            self.add_flow(inference, post_inference_op, {("", "in")})
            self.add_flow(post_inference_op, viz, {("out", "receivers")})
            self.add_flow(post_inference_op, viz, {("output_specs", "input_specs")})

        else:
            print(f"plesea either choose v4l2 or replayer")


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument(
        "-c",
        "--config",
        default="ready.yaml",
        help=("configuration file (e.g. ready.yaml)"),
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "v4l2"],
        default="replayer",
        help=("If 'replayer', replay a prerecorded video. If 'v4l2' use an v4l2"),
    )
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
    parser.add_argument(
        "-df",
        "--debug_print_flag",
        type=lambda s: s.lower() in ["true", "t", "yes", "1"],
        default=True,
        help=(
            "Set debug flag either False or True (default). \
                WARNING: Setting this to True will slow down performance of the app!"
        ),
    )
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), args.config)

    app = READYApp(
        source=args.source,
        data=args.data,
        model_name=args.model_name,
        debug_print_flag=args.debug_print_flag,
    )

    with Tracker(app, filename=args.logger_filename) as tracker:
        app.config(config_file)
        app.run()
