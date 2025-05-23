import argparse
import asyncio
import json
import logging
import os
import ssl
from threading import Condition, Event, Thread

import cupy as cp
import holoscan
import holoscan as hs
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamError, MediaStreamTrack
from holoscan import as_tensor
from holoscan.conditions import PeriodicCondition
from holoscan.core import (Application, ConditionType, IOSpec, Operator,
                           OperatorSpec, Tracker)
from holoscan.gxf import Entity
from holoscan.operators import (FormatConverterOp, HolovizOp, InferenceOp,
                                SegmentationPostprocessorOp,
                                VideoStreamRecorderOp, VideoStreamReplayerOp)
from holoscan.resources import (BlockMemoryPool, CudaStreamPool,
                                MemoryStorageType, UnboundedAllocator)
from holoscan.schedulers import (EventBasedScheduler, GreedyScheduler,
                                 MultiThreadScheduler)

ROOT = os.path.dirname(__file__)


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
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        # print(f" >>> Start Compute PreInfoOp  ")
        in_message = op_input.receive("in")
        # print(in_message)
        tensor = cp.asarray(in_message.get("frame"), dtype=cp.float32)
        # tensor_1ch = tensor[:, :, 0]
        # print(f"tensor.shape={tensor.shape}")
        # print(f"tensor_1ch.shape={tensor_1ch.shape}")

        out_message = Entity(context)
        # out_message.add(hs.as_tensor(tensor_1ch), "tensor1ch")
        out_message.add(hs.as_tensor(tensor), "tensor")
        op_output.emit(in_message, "out")
        # print(f" >>> End Compute PreInfoOp  ")


class PostInferenceOp(Operator):
    """
    Information Operator

    Input:
        in: source_video

    Output:
        out: 'dynamic_text': array([[ x_coord, y_coord, font_size)  ])
        output_specs: holoscan.operators.holoviz._holoviz.HolovizOp.InputSpec object
    """

    def __init__(self, *args, **kwargs):
        """Initialize Operator"""
        self.frame_count = 0
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator"""
        spec.input("in")
        spec.output("outputs")
        spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        # print(f"--START-------- PostInferenceOp  ------------")
        in_message = op_input.receive("in")
        # print(f"in_message={in_message}")
        # print(f"frame count={self.frame_count}")
        tensor = cp.asarray(in_message.get("unet_out"), dtype=cp.float32)
        # print(f"unet_out tensor.shape={tensor.shape}") #tensor.shape=(1, 4, 400, 640)
        # print(f"tensor.dtype={tensor.dtype}")
        # print(f"tensor.min()={cp.min(tensor)}")
        # print(f"tensor.max()={cp.max(tensor)}")
        # print(f"tensor.mean()={cp.mean(tensor)}")

        dynamic_text_coord = cp.asarray(
            [
                (0.01, 0.01, 0.035),  # (x, y, font_size)
            ],
        )
        out_message = {
            "dynamic_text_coord": dynamic_text_coord,
        }
        # print(f"out_message: {out_message}")
        op_output.emit(out_message, "outputs")

        specs = []
        spec = HolovizOp.InputSpec("dynamic_text_coord", HolovizOp.InputType.TEXT)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.width = 1.25
        view.height = 1.25

        spec.text = [
            "Frame "
            + str(self.frame_count)
            # + " tensor.min()="
            # + str(cp.min(tensor))
            # + " tensor.max()="
            # + str(cp.max(tensor))
            # + " tensor.mean()="
            # + str(cp.mean(tensor))
        ]
        spec.color = [
            1,
            1,
            1,
            0.99,
        ]
        spec.priority = 1

        spec.views = [view]
        specs.append(spec)
        # print(f"specs: {specs}")
        op_output.emit(specs, "output_specs")

        self.frame_count += 1
        # print(f"--END-------- PostInferenceOp  ------------")


class DropFramesOp(Operator):
    """Dropping Frames Operator"""
    def __init__(self, fragment, *args, **kwargs):
        """Initialize Operator
        Need to call the base class constructor last
        """
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setting up specifications of Operator

        Notes:
        ---------
        For policy:
          - IOSpec.QueuePolicy.POP = pop the oldest value in favor of the new one when the queue
            is full
          - IOSpec.QueuePolicy.REJECT = reject the new value when the queue is full
          - IOSpec.QueuePolicy.FAULT = fault if queue is full (default)

        One could also set the receiver's capacity and policy via the connector method:
            .connector(
                IOSpec.ConnectorType.DOUBLE_BUFFER,
                capacity=1,
                policy=1,  # 1 = reject, 0 = pop, 2 = fault
            )
        but that is less flexible as `IOSpec::ConnectorType::kDoubleBuffer` is appropriate for
        within-fragment connections, but will not work if the operator was connected to a
        different fragment.

        """
        spec.input("in", policy=IOSpec.QueuePolicy.REJECT).condition(
            ConditionType.MESSAGE_AVAILABLE,
            min_size=1,
            front_stage_max_size=1
        )
        # spec.input("in").connector(
        #     IOSpec.ConnectorType.DOUBLE_BUFFER,
        #     capacity=1,
        #     policy=1,
        # ).condition(ConditionType.MESSAGE_AVAILABLE, min_size=1, front_stage_max_size=1)
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Computing method to receive input message and emit output message"""
        value = op_input.receive("in")
        op_output.emit(value, "out")


class VideoStreamReceiverContext:
    def __init__(self):
        self.task = None


class VideoStreamReceiver:
    """
    Video Stream Receiver
    https://github.com/nvidia-holoscan/holohub/blob/main/operators/webrtc_client/webrtc_client_op.py
    """
    def __init__(self, video_frame_available: Condition, video_frames: list):
        self._video_frame_available = video_frame_available
        self._video_frames = video_frames
        self._receiver_contexts = {}

    def add_track(self, track: MediaStreamTrack):
        context = VideoStreamReceiverContext()
        context.task = asyncio.ensure_future(self._run_track(track, context))
        self._receiver_contexts[track] = context

    def remove_track(self, track: MediaStreamTrack):
        context = self._receiver_contexts[track]
        if context:
            if context.task is not None:
                context.task.cancel()
                context.task = None
            self._receiver_contexts.pop(track)

    async def stop(self):
        for context in self._receiver_contexts:
            if context.task is not None:
                context.task.cancel()
                context.task = None
        self._receiver_contexts.clear()

    async def _run_track(self, track: MediaStreamTrack, context: VideoStreamReceiverContext):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return

            with self._video_frame_available:
                self._video_frames.append(frame)
                self._video_frame_available.notify_all()


class WebRTCClientOp(Operator):
    """
    WebRTC Client Operator
    The webrtc_client operator receives video frames through a WebRTC connection.
    The application using this operator needs to call the offer method of the operator when a new WebRTC connection is available.

    Methods

    Start a connection between the local computer and the peer.
    async def offer(self, sdp, type) -> (local_sdp, local_type)

    Parameters
        sdp peer Session Description Protocol object
        type peer session type
        Return values

        sdp local Session Description Protocol object
        type local session type

    Outputs
        output: Tensor with 8 bit per component RGB data
        type: Tensor

    https://github.com/nvidia-holoscan/holohub/blob/main/operators/webrtc_client/webrtc_client_op.py
    """
    def __init__(self, *args, **kwargs):
        self._connected = False
        self._connected_event = Event()
        self._video_frame_available = Condition()
        self._video_frames = []
        self._pcs = set()
        self._receiver = VideoStreamReceiver(self._video_frame_available, self._video_frames)
        super().__init__(*args, **kwargs)

    async def offer(self, sdp, type):
        offer = RTCSessionDescription(sdp, type)

        pc = RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"Connection state {pc.connectionState}")
            if pc.connectionState == "connected":
                self._connected = True
                self._connected_event.set()
            elif pc.connectionState == "failed":
                await pc.close()
                self._pcs.discard(pc)
                self._connected = False
                self._connected_event.set()

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self._receiver.add_track(track)

            @track.on("ended")
            async def on_ended():
                self._receiver.remove_track(track)

        # handle offer
        await pc.setRemoteDescription(offer)

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return (pc.localDescription.sdp, pc.localDescription.type)

    async def shutdown(self):
        # close peer connections
        coros = [pc.close() for pc in self._pcs]
        await asyncio.gather(*coros)
        self._pcs.clear()

    def setup(self, spec: OperatorSpec):
        # Note: Setting ConditionType.NONE overrides the default of
        #   ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE. This means that the operator will be
        #   triggered regardless of whether any operators connected downstream have space in their
        #   queues.
        # spec.output("output")
        spec.output("output").condition(ConditionType.NONE)

    def start(self):
        self._connected_event.wait()
        if not self._connected:
            exit(-1)

    def stop(self):
        self._receiver.stop()

    def compute(self, op_input, op_output, context):
        video_frame = None
        with self._video_frame_available:
            while not self._video_frames:
                self._video_frame_available.wait()
            video_frame = self._video_frames.pop(0)

        rgb_frame = video_frame.to_rgb()
        array = rgb_frame.to_ndarray()

        entity = Entity(context)
        entity.add(as_tensor(array), "frame")
        op_output.emit(entity, "output")


class WebAppThread(Thread):
    def __init__(self, webrtc_client_op, host, port, cert_file=None, key_file=None):
        super().__init__()
        self._webrtc_client_op = webrtc_client_op
        self._host = host
        self._port = port

        if cert_file:
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._ssl_context.load_cert_chain(cert_file, key_file)
        else:
            self._ssl_context = None

        app = web.Application()
        app.on_shutdown.append(self._on_shutdown)
        app.router.add_get("/", self._index)
        app.router.add_get("/client.js", self._javascript)
        app.router.add_post("/offer", self._offer)

        self._runner = web.AppRunner(app)

    async def _on_shutdown(self, app):
        self._webrtc_client_op.shutdown()

    async def _index(self, request):
        content = open(os.path.join(ROOT, "index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)

    async def _javascript(self, request):
        content = open(os.path.join(ROOT, "client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

    async def _offer(self, request):
        params = await request.json()

        (sdp, type) = await self._webrtc_client_op.offer(params["sdp"], params["type"])

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": sdp, "type": type}),
        )

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, self._host, self._port, ssl_context=self._ssl_context)
        logging.info(f"Starting web server at {self._host}:{self._port}")
        loop.run_until_complete(site.start())
        loop.run_forever()


class WebRTCClientApp(Application):
    def __init__(self, cmdline_args):
        super().__init__()
        self._cmdline_args = cmdline_args

    def compose(self):
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        webrtc_client_op = WebRTCClientOp(self, name="WebRTC Client")
        visualizer_sink = HolovizOp(
            self,
            name="HolovizOp_sink",
            window_title="READY Demo WebRTC Client",
            width=640, #320 #TODO pass this as a width and height from index.html video-resolution
            height=480, #240
            cuda_stream_pool=cuda_stream_pool,
            tensors=[
                dict(
                    name="frame",
                    type="color",
                    priority=0,
                    opacity=1.0,
                    image_format="r8g8b8_unorm", #r8g8b8_snorm #r8g8b8_srgb
                ),
                dict(
                    name="out_tensor",
                    type="color_lut",
                    priority=0,
                    opacity=1.0,
                ),
            ],
            color_lut=[
                [0.65, 0.81, 0.89, 0.01], #background #RGB for light blue & alpha=0.1
                [0.3, 0.3, 0.9, 0.5], #sclera  #RGB for blue & alpha=0.5
                [0.1, 0.8, 0.2, 0.5], #Iris    #RGB for green & alpha=0.5
                [0.9, 0.9, 0.3, 0.8], #Pupil   #RGB for yellow & alpha=0.8
                #https://rgbcolorpicker.com/0-1
            ],
            enable_render_buffer_input=False, #default: `false`
            enable_render_buffer_output=False, #default: `false` #TODO self._cmdline_args.enable_recording
        )


        pre_info_op = PreInfoOp(
            self,
            name="pre_info_op",
            allocator=UnboundedAllocator(self, name="host_allocator")
        )


        post_inference_op = PostInferenceOp(
            self,
            name="post_inference_op",
            allocator=host_allocator,
        )


        model_width = 640
        model_height = 400
        n_channels_inference = 4
        bpp_inference = 4
        inference_block_size = model_width * model_height * n_channels_inference * bpp_inference
        inference_num_blocks = 2

        formatter_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        bytes_per_float32 =4
        in_components=3
        format_op = FormatConverterOp(
            self,
            name="format_op",
            in_dtype="rgb888", #"rgba8888" for four channels; float32" for 3 channels
            out_dtype="float32", #"rgb888",float32 for 3 channels
            pool=BlockMemoryPool(
                self,
                name="format_op_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=model_width * model_height * bytes_per_float32 * in_components,
                num_blocks=2*3,
            ),
            # pool=UnboundedAllocator(self, name="FormatConverterOp allocator"),
            in_tensor_name="frame",
            out_tensor_name="out_format_op",
            scale_min=1.0,
            scale_max=252.0,
            alpha_value=255,
            resize_width=model_width,
            resize_height=model_height,
            cuda_stream_pool=formatter_cuda_stream_pool,
        )

        self_models_path_map = {
            "ready_model": os.path.join(self._cmdline_args.models_path_map, self._cmdline_args.model_name),
        }

        inference_op = InferenceOp(
            self,
            name="segmentation_inference_op",
            backend="trt",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=inference_block_size,
                num_blocks=inference_num_blocks,
            ),
            model_path_map=self_models_path_map,
            pre_processor_map={"ready_model": ["out_format_op"]},
            inference_map={"ready_model": "unet_out"},
            enable_fp16=False, #Use 16-bit floating point computations. Optional (default: `False`).
            parallel_inference=True, # optional param, default to True
            infer_on_cpu=False, # optional param, default to False
            input_on_cuda=True, # optional param, default to True
            output_on_cuda=True, # optional param, default to True
            transmit_on_cuda=True, # optional param, default to True
            is_engine_path=False, # optional param, default to False
        )

        segpostprocessor_op = SegmentationPostprocessorOp(
            self,
            name="segpostprocessor",
            allocator=UnboundedAllocator(self, name="segpostprocessor_allocator"),
            in_tensor_name="unet_out",
            network_output_type="softmax", #softmax layer for multiclass segmentation  #sigmoid layer for binary segmentation
            data_format="nchw",
        )

        branch_hz = 15
        period_ns = int(1e9 / branch_hz)
        drop_frames_op = DropFramesOp(
            self,
            PeriodicCondition(self, recess_period=period_ns),
            name="drop_frames_op",
        )

        replayer_op = VideoStreamReplayerOp(
            self,
            name="replayer_op",
            directory=self._cmdline_args.recording_directory,
            basename=self._cmdline_args.recording_basename,
            frame_rate=0,
            repeat=True, # default: false
            realtime=True, # default: true
            count=0, # default: 0 (no frame count restriction)
        )

        recorder_op = VideoStreamRecorderOp(
            name="recorder_op",
            fragment=self,
            directory=self._cmdline_args.recording_directory,
            basename=self._cmdline_args.recording_basename,
        )
        visualizer_replayer = HolovizOp(
            self,
            name="Video Replayer Sink",
            window_title="Replayer WebRTC Client",
            width=640, #TODO pass this as a width and height from index.html video-resolution
            height=480,
            cuda_stream_pool=cuda_stream_pool,
            tensors=[
                dict(
                    name="frame",
                    type="color",
                    priority=0,
                    opacity=1.0,
                    image_format="r8g8b8_unorm", #r8g8b8_snorm #r8g8b8_srgb
                ),
            ],
            enable_render_buffer_input=False, #default: `false`
            enable_render_buffer_output=False, #default: `false` #TODO self._cmdline_args.enable_recording
        )

	## WORKFLOW
        ### WebRTC
        if self._cmdline_args.source == "webrtc":
            ### Branch01
            self.add_flow(webrtc_client_op, drop_frames_op, {("output", "in")})
            self.add_flow(drop_frames_op, visualizer_sink, {("out", "receivers")})

	        ### Branch02
            self.add_flow(webrtc_client_op, drop_frames_op, {("output", "in")})
            self.add_flow(drop_frames_op, pre_info_op, {("out", "in")})
            self.add_flow(pre_info_op, format_op, {("out", "")})
            self.add_flow(format_op, inference_op, {("tensor", "receivers")})

            self.add_flow(inference_op, segpostprocessor_op, {("transmitter", "")})
            self.add_flow(segpostprocessor_op, visualizer_sink, {("", "receivers")})

            self.add_flow(inference_op, post_inference_op, {("", "in")})

            self.add_flow(post_inference_op, visualizer_sink, {("outputs", "receivers")})
            self.add_flow(post_inference_op, visualizer_sink, {("output_specs", "input_specs")})


        ### Recorder
        if self._cmdline_args.enable_recording == "True":
            self.add_flow(webrtc_client_op, recorder_op, {("output", "input")})
            #TODO: # if record_type == "input":  elif record_type == "visualizer":
            #TODO self.add_flow(visualizer_sink, recorder_op, {("render_buffer_output", "input")})


        ### Replayer Raw Video
        if self._cmdline_args.source == "replayer_raw":
            self.add_flow(replayer_op, visualizer_replayer, {("output", "receivers")})


        ### Replayer Inference Video
        if self._cmdline_args.source == "replayer_inference":
            ### Branch01
            self.add_flow(replayer_op, visualizer_sink, {("output", "receivers")})

            ### Branch02
            self.add_flow(replayer_op, format_op, {("output", "")})
            self.add_flow(format_op, inference_op, {("tensor", "receivers")})

            self.add_flow(inference_op, segpostprocessor_op, {("transmitter", "")})
            self.add_flow(segpostprocessor_op, visualizer_sink, {("", "receivers")})

            self.add_flow(inference_op, post_inference_op, {("", "in")})

            self.add_flow(post_inference_op, visualizer_sink, {("outputs", "receivers")})
            self.add_flow(post_inference_op, visualizer_sink, {("output_specs", "input_specs")})

        ## REFERENCE
        ## self.add_flow(upstreamOP, downstreamOP, {("output_portname_upstreamOP", "input_portname_downstreamOP")})

        # start the web server in the background, this will call the WebRTC server operator
        # 'offer' method when a connection is established
        self._web_app_thread = WebAppThread(
            webrtc_client_op,
            self._cmdline_args.host,
            self._cmdline_args.port,
            self._cmdline_args.cert_file,
            self._cmdline_args.key_file,
        )
        self._web_app_thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "-l",
        "--logger_filename",
        default="logger.log",
        help=("Set logger filename"),
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default="_weights_15-12-24_07-00-10-sim-BHWC.onnx",
        help=("Set model name"),
    )
    parser.add_argument(
        "-mp",
        "--models_path_map",
        default="/workspace/volumes/datasets/ready/mobious/models_a10080gb/15-12-24",
        help=("Set model path"),
    )
    parser.add_argument(
        "-s",
        "--source",
        default="webrtc",
        choices=[
            "webrtc",
            "replayer_raw",
            "replayer_inference",
        ],
        help="Source of the video stream (default: webrtc)",
    )
    parser.add_argument(
        "-r",
        "--enable_recording",
        default=False,
        help="Enable recording of the video stream (default: False)",
    )
    parser.add_argument(
        "-rd",
        "--recording_directory",
        help=("Set recording directory"),
    )
    parser.add_argument(
        "-rb",
        "--recording_basename",
        help=("Set recording basename"),
    )
    cmdline_args = parser.parse_args()

    if cmdline_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = WebRTCClientApp(cmdline_args)


    # Experimenting to improve consumer speed with schedulers
    scheduler = GreedyScheduler(app, name="greedy_scheduler") # Default scheduler for thread = 0 ;
    # scheduler_class = EventBasedScheduler
    # scheduler_class = MultiThreadScheduler
    # scheduler = scheduler_class(
    #     app,
    #     worker_thread_number=5,
    #     stop_on_deadlock=True,
    #     stop_on_deadlock_timeout=500,
    #     name="webrtc_scheduler",
    # )
    app.scheduler(scheduler)

    with Tracker(app, filename=cmdline_args.logger_filename, num_start_messages_to_skip=2, num_last_messages_to_discard=2) as tracker:
        app.run()
