import os
import cupy as cp
import cv2
import holoscan as hs

from argparse import ArgumentParser

from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.gxf import Entity
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
    V4L2VideoCaptureOp,
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, UnboundedAllocator


class PreInfoOp(Operator):
    """
    Pre Information Operator

    Input:

    Output:
    """

    def __init__(self, *args, **kwargs):
        """Initialize Operator"""
        #self.frame_count = 1
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("source_video")
        spec.output("out")
        # spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        print(f"---------- PreInfoOp  ------------")
        in_message = op_input.receive("source_video")
        tensor = cp.asarray(in_message.get(""), dtype=cp.float32)
        tensor_1ch =  tensor[:,:,0]
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
        #self.frame_count = 1
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        print(f"********** FormatInferenceInputOp  ************")
        in_message = op_input.receive("in")
        #print(in_message)
        tensor = cp.asarray(in_message.get("out_preprocessor"), dtype=cp.float32)
        print(f"**") 
        print(f"tensor.shape={tensor.shape}")#tensor.shape=(400, 640, 3)

        print(f"tensor.min {cp.min(tensor)}")
        print(f"tensor.max {cp.max(tensor)}")
        print(f"tensor.mean {cp.mean(tensor)}")

        tensor_ = cp.moveaxis(tensor, 2, 0)[None]
        print(f"**") 
        print(f"tensor_.shape={tensor_.shape}") #tensor_.shape=(1, 3, 400, 640)
        print(f"tensor_.min {cp.min(tensor_)}")
        print(f"tensor_.max {cp.max(tensor_)}")
        print(f"tensor_.mean {cp.mean(tensor_)}")

        tensor_1ch =  tensor_[:,0,:,:]
        print(f"**") 
        print(f"tensor_1ch.shape={tensor_1ch.shape}")#tensor_1ch.shape=(1, 400, 640)
        print(f"tensor1ch.min {cp.min(tensor_1ch)}")
        print(f"tensor1ch.max {cp.max(tensor_1ch)}")
        print(f"tensor1ch.mean {cp.mean(tensor_1ch)}")

        tensor_1CH = cp.expand_dims(tensor_1ch, 0)
        print(f"**") 
        print(f"tensor_1CH.shape={tensor_1CH.shape}") #tensor_1CH.shape=(1, 1, 400, 640)
        print(f"tensor_1CH.min {cp.min(tensor_1CH)}")
        print(f"tensor_1CH.max {cp.max(tensor_1CH)}")
        print(f"tensor_1CH.mean {cp.mean(tensor_1CH)}")

        tensor_1Ch = cp.expand_dims(tensor_1ch, -1)
        print(f"**") 
        print(f"tensor_1Ch.shape={tensor_1Ch.shape}") #tensor_1CH.shape=(1, 400, 640, 1)
        print(f"tensor_1Ch.min {cp.min(tensor_1Ch)}")
        print(f"tensor_1Ch.max {cp.max(tensor_1Ch)}")
        print(f"tensor_1Ch.mean {cp.mean(tensor_1Ch)}")
	
        out_message = Entity(context)
        out_message.add(hs.as_tensor(tensor_), "out_preprocessor")
        #out_message.add(hs.as_tensor(tensor_1Ch), "out_preprocessor")
        op_output.emit(out_message, "out")

def cv_cuda_gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    print("::: cv_cuda_gpumat_from_cp_array :::")
    print(arr.__cuda_array_interface__['shape'])
    print(arr.__cuda_array_interface__['data'][0])
    type_map = {
        cp.dtype('uint8'): cv2.CV_8U,
        cp.dtype('int8'): cv2.CV_8S,
        cp.dtype('uint16'): cv2.CV_16U,
        cp.dtype('int16'): cv2.CV_16S,
        cp.dtype('int32'): cv2.CV_32S,
        cp.dtype('float32'): cv2.CV_32F,
        cp.dtype('float64'): cv2.CV_64F
    }
    print(f'arr.dtype: {arr.dtype}') #float32
    depth = type_map.get(arr.dtype)
    assert depth is not None, "Unsupported CuPy array dtype"
    print(f' depth: {depth}')
    channels = 1 if len(arr.shape) == 2 else arr.shape[1]
    mat_type = depth + ((channels - 1) << 3)
    print(mat_type)

    mat = cv2.cuda.createGpuMatFromCudaMemory(
        arr.__cuda_array_interface__["shape"],
        mat_type,
        arr.__cuda_array_interface__["data"][0],
    )
    #print(type(mat)) #<class 'cv2.cuda.GpuMat'>
    print(mat.size()) #(400, 640)
    print(mat.channels()) #1
    print("::: cv_cuda_gpumat_from_cp_array :::")
    return mat


class PostInferenceOp(Operator):
    """
    Post Inference Operator

    Input:

    Output:
    """

    #def __init__(self, *args, **kwargs):
    def __init__(self, fragment, width=None, height=None,  **kwargs):
        """Initialize Operator"""
        super().__init__(fragment, **kwargs)
        self.frame_count = 1
        self.width=width
        self.height=height
        self.centroid_coords = cp.array([0])
        self.cycle = 0
        self.x = cp.linspace(0, 1.0, 30)
        self.varing_array = cp.zeros((30, 2))


    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("in")
        spec.output("out")
        spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        print(f"----------------------------------------------")
        print(f"---------- PostInferenceOperator  ------------")
        in_message = op_input.receive("in")
        #print(f"in_message={in_message}")
        tensor = cp.asarray(in_message.get("unet_out"), dtype=cp.float32)
        print(f"unet_out tensor.shape={tensor.shape}") #tensor.shape=(1, 4, 400, 640)

        tensor_1ch_background =  tensor[:,0,:,:]
        tensor_1ch_sclera =  tensor[:,1,:,:]
        tensor_1ch_iris =  tensor[:,2,:,:]

        ### CENTROID OF PUPIL MASK
        tensor_1ch_pupil =  tensor[:,3,:,:]
        print(f"tensor.min {cp.min(tensor_1ch_pupil)}")
        print(f"tensor.max {cp.max(tensor_1ch_pupil)}")
        print(f"tensor.mean {cp.mean(tensor_1ch_pupil)}")

        print(f"tensor_1ch_pupil.shape={tensor_1ch_pupil.shape}") #tensor.shape=(1, 400, 640)
        tensor_1ch_pupil_sq = cp.squeeze(tensor_1ch_pupil, axis=None)
        print(f"tensor_1ch_pupil_sq.shape={tensor_1ch_pupil_sq.shape}") #tensor.shape=(400, 640)
        tensor_1ch_pupil_sq_uint8 = tensor_1ch_pupil_sq.astype(cp.uint8)
        print(tensor_1ch_pupil_sq_uint8.dtype) #uint8

        mask_pupil_bool = tensor_1ch_pupil_sq_uint8 > 1
        print(f"mask_pupil_bool.shape {mask_pupil_bool.shape}") #tensor.shape=(1, 4, 400, 640)
        print(f"mask_pupil_bool.dtype {mask_pupil_bool.dtype}") #bool

        centroid = cp.mean(cp.argwhere(mask_pupil_bool),axis=0)
        centroid = cp.nan_to_num(centroid) #convert float NaN to integer
        print(f"centroid: {centroid}")
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
        #https://stackoverflow.com/questions/73131778/

#	#EXPERIMENTAL (to be removed or checked for multiple mask)
#        #https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
#        frameUMat = cv2.UMat(tensor_1ch_pupil_sq_uint8.shape[0], tensor_1ch_pupil_sq_uint8.shape[1], cv2.CV_8U)
#        mask_gray = cv2.normalize(src=frameUMat, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
#        #print(mask_gray.get())
#        blur = cv2.GaussianBlur(mask_gray, (5, 5), 0)
#        ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
#        # calculate x,y coordinate of center
#        M = cv2.moments(thresh)
#        #M = cv2.cuda.moments(thresh)
#        #mulitple objects
#        #contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
                (centroid_x/self.width, centroid_y/self.height),
            ],
            dtype=cp.float32,
        )
        #print(centroid_xy) #[[320. 200.]]
        centroid_xy = centroid_xy[cp.newaxis, :, :]
        #print(centroid_xy) #[[[320. 200.]]]

        out_message = Entity(context)

        text_coords = cp.asarray(
            [
                #(0.01, 0.01, 0.03),  # (x, y, font_size)
                (centroid_x/self.width, centroid_y/self.height, 0.03),
            ],
            dtype=cp.float32
        )
        text_coords = text_coords[cp.newaxis, :, :]



        #######################################
        # Create a time-varying "points" tensor
        #######################################
        # Set of (x, y) points with 30 points equally spaced along x 
        # whose y coordinate varies based on self.centroid_coords over time.
        #self.centroid_coords = cp.append(self.centroid_coords, centroid_x/self.width)
        self.centroid_coords = cp.append(self.centroid_coords, centroid_y/self.height)
        if self.frame_count%29 == 0:
            #print(self.centroid_coords)
            self.varing_array = cp.stack((self.x, self.centroid_coords), axis=-1) 
            print(self.varing_array)
            #print(self.varing_array.shape)
            self.cycle = 0
            self.centroid_coords = cp.array([0])

        self.cycle += 1

        y = 0.8 + 0.1 * cp.sin(8 * cp.pi * self.x + self.frame_count / 60 * 2 * cp.pi)
        point_coords = cp.stack((self.x, y), axis=-1) #Stack so the final shape is (n_points, 2)


        #adds messages
        out_message.add(hs.as_tensor(centroid_xy), "pupil_cXcY")
        out_message.add(hs.as_tensor(text_coords), "text_coords")
        #out_message.add(hs.as_tensor(point_coords), "point_coords")
        out_message.add(hs.as_tensor(self.varing_array), "point_coords")
        op_output.emit(out_message, "out")

        specs = []
        spec = HolovizOp.InputSpec("text_coords", HolovizOp.InputType.TEXT)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.02
        view.offset_y = 0.02
        view.width = 1.0
        view.height = 1.0

        spec.text = [
            "Frame " + str(self.frame_count) + "\n"
            + " unetout shape=" + str(tensor.shape) + "\n"
            + " t.min="
            + str(cp.min(tensor))
            + " t.max="
            + str(cp.max(tensor))
            + " t.mean="
            + str(cp.mean(tensor)) + "\n"
            + " cx="
            + str(centroid_x/self.width)
            + " cy="
            + str(centroid_y/self.height)
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
        op_output.emit(specs, "output_specs")

        self.frame_count += 1


class READYApp(Application):
    def __init__(self, source=None, data=None, model_name=None, debug_print_flag=None, **kwargs):
        """Initialize the application

        Parameters
        ----------
        data : Location to the data
        model_name : Model name
        """

        super().__init__(**kwargs)

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
            width = 640 #TODO source_args["width"]
            height = 400 #TODO source_args["height"]
            #n_channels = 4  # RGBA
            n_channels = 1
            bpp = 4  # bytes per pixel
            block_size = width * height * n_channels
            drop_alpha_block_size = width * height * n_channels * bpp
            drop_alpha_num_blocks = 2
            allocator = BlockMemoryPool(
                self,
                name="pool",
                storage_type=0,  # storage_type=MemoryStorageType.DEVICE,
                block_size=drop_alpha_block_size,
                num_blocks=drop_alpha_num_blocks,
            )
            source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.video_dir,
            **self.kwargs("replayer"),
            )


        elif self.source.lower() == "v4l2":
            width = source_args["width"]
            height = source_args["height"]
            n_channels = 4  # RGBA
            bpp = 4  # bytes per pixel
            block_size = width * height * n_channels
            drop_alpha_block_size = width * height * n_channels * bpp
            drop_alpha_num_blocks = 2
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

        else:
            print(f"plesea either choose v4l2 or replayer")


        cuda_stream_pool = CudaStreamPool(
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
            name="pre_info_op",
            allocator=host_allocator,
            **self.kwargs("pre_info_op"),
        )

        preprocessor_replayer = FormatConverterOp(
            self, name="preprocessor", 
            pool=host_allocator, 
            #pool=allocator, 
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("preprocessor_replayer"),
        )

        preprocessor_v4l2 = FormatConverterOp(
            self,
            name="preprocessor_v4l2",
            pool=host_allocator,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("preprocessor_v4l2"),
        )

        format_input = FormatInferenceInputOp(
            self,
            name="format_input",
            allocator=host_allocator,
            **self.kwargs("format_input"),
        )

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=self.models_path_map,
            **self.kwargs("inference"),
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
            **self.kwargs("segpostprocessor"),
        )

        viz = HolovizOp(
            self, 
            name="viz", 
            cuda_stream_pool=cuda_stream_pool, 
            **self.kwargs("viz"),
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
            "Set debug flag either False or True (default). WARNING: Setting this to True will slow down performance of the app!"
        ),
    )
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "ready.yaml")

    app = READYApp(
        source=args.source,
        data=args.data, 
        model_name=args.model_name,
        debug_print_flag=args.debug_print_flag,
    )
    with Tracker(app, filename=args.logger_filename) as tracker:
       app.config(config_file)
       app.run()
