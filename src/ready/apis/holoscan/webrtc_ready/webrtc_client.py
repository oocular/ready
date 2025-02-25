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
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.gxf import Entity
from holoscan.operators import (FormatConverterOp, HolovizOp, InferenceOp,
                                SegmentationPostprocessorOp)
from holoscan.resources import (BlockMemoryPool, CudaStreamPool,
                                MemoryStorageType, UnboundedAllocator)

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
        print(f" >>> Start Compute PreInfoOp  ")
        in_message = op_input.receive("in")
        print(in_message)
        tensor = cp.asarray(in_message.get("frame"), dtype=cp.float32)
        tensor_1ch = tensor[:, :, 0]
        print(f"tensor.shape={tensor.shape}")
        print(f"tensor_1ch.shape={tensor_1ch.shape}")

        out_message = Entity(context)
        # out_message.add(hs.as_tensor(tensor_1ch), "tensor1ch")
        out_message.add(hs.as_tensor(tensor), "tensor")
        op_output.emit(in_message, "out")
        print(f" >>> End Compute PreInfoOp  ")


class InfoOp(Operator):
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
        print(f"---------- InfoOp  ------------")
        in_message = op_input.receive("in")
        print(f"in_message={in_message}")
        print(f"frame count={self.frame_count}")
        # tensor = cp.asarray(in_message.get("frame"))#tensor.dtype=uint8
        tensor = cp.asarray(in_message.get("frame"), dtype=cp.float32) #tensor.dtype=float32
        print(f"tensor.shape={tensor.shape}")
        print(f"tensor.dtype={tensor.dtype}")
        print(f"tensor.min()={cp.min(tensor)}")
        print(f"tensor.max()={cp.max(tensor)}")
        print(f"tensor.mean()={cp.mean(tensor)}")

        dynamic_text_coord = cp.asarray(
            [
                (0.01, 0.01, 0.035),  # (x, y, font_size)
            ],
        )
        out_message = {
            "dynamic_text_coord": dynamic_text_coord,
        }
        print(f"out_message: {out_message}")
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
            0.99,
        ]
        spec.priority = 1

        spec.views = [view]
        specs.append(spec)
        print(f"specs: {specs}")
        op_output.emit(specs, "output_specs")

        self.frame_count += 1



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
        spec.output("output")

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
            name="Video Sink",
            window_title="WebRTC Client",
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
        )
        info_op = InfoOp(
            self,
            name="info_op",
            allocator=host_allocator,
        )

        models_path_map="/workspace/volumes/datasets/ready/mobious/models"
        model_name="_weights_15-12-24_07-00-10-sim-BHWC.onnx"
        self_models_path_map = {
            "ready_model": os.path.join(models_path_map, model_name),
        }

        model_width = 640
        model_height = 400
        n_channels_inference = 4
        bpp_inference = 4
        inference_block_size = model_width * model_height * n_channels_inference * bpp_inference
        inference_num_blocks = 2

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
            pre_processor_map={"ready_model": ["out_preprocessor"]},
            inference_map={"ready_model": "unet_out"},
            enable_fp16=False, #Use 16-bit floating point computations. Optional (default: `False`).
            parallel_inference=True, # optional param, default to True
            infer_on_cpu=False, # optional param, default to False
            input_on_cuda=True, # optional param, default to True
            output_on_cuda=True, # optional param, default to True
            transmit_on_cuda=True, # optional param, default to True
            is_engine_path=False, # optional param, default to False
        )


        pre_info_op = PreInfoOp(
            self,
            name="pre_info_op",
            allocator=UnboundedAllocator(self, name="host_allocator")
        )


        segpostprocessor_op = SegmentationPostprocessorOp(
            self,
            name="segpostprocessor",
            allocator=UnboundedAllocator(self, name="segpostprocessor_allocator"),
            in_tensor_name="unet_out",
            network_output_type="softmax", #softmax layer for multiclass segmentation  #sigmoid layer for binary segmentation
            data_format="nchw",
        )

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
            in_dtype="float32", #"rgba8888" for four channels; float32" for 3 channels
            out_dtype="float32", #"rgb888",float32 for 3 channels
            pool=BlockMemoryPool(
                self,
                name="format_op_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=model_width * model_height * bytes_per_float32 * in_components,
                num_blocks=2*3,
            ),
            # pool=UnboundedAllocator(self, name="FormatConverterOp allocator"),
            out_tensor_name="out_format_op",
            scale_min=1.0,
            scale_max=252.0,
            alpha_value=255,
            resize_width=model_width,
            resize_height=model_height,
            cuda_stream_pool=formatter_cuda_stream_pool,
        )

        ## REFERENCE
        ## self.add_flow(upstreamOP, downstreamOP, {("output_portname_upstreamOP", "input_portname_downstreamOP")})
        #>>>>>>>>>>>>>>>>>
        ##WORFLOW1 WORKS
        # self.add_flow(webrtc_client_op, visualizer_sink, {("output", "receivers")})
        # self.add_flow(webrtc_client_op, info_op, {("", "in")})
        # self.add_flow(info_op, visualizer_sink, {("outputs", "receivers")})
        # self.add_flow(info_op, visualizer_sink, {("output_specs", "input_specs")})
        #>>>>>>>>>>>>>>>>>

        #>>>>>>>>>>>>>>>>>
        ##WORKFLOW2 WORKS
        # self.add_flow(webrtc_client_op, visualizer_sink, {("output", "receivers")})
        # self.add_flow(webrtc_client_op, pre_info_op, {("output", "in")})
        # self.add_flow(pre_info_op, visualizer_sink, {("out", "receivers")})
        #>>>>>>>>>>>>>>>>>

        #>>>>>>>>>>>>>>>>>
        ##WORKFLOW3
        self.add_flow(webrtc_client_op, visualizer_sink, {("output", "receivers")})
        self.add_flow(webrtc_client_op, pre_info_op, {("output", "in")})
        self.add_flow(pre_info_op, format_op, {("out", "")})
        self.add_flow(format_op, inference_op, {("tensor", "")})
        self.add_flow(inference_op, segpostprocessor_op, {("transmitter", "")})
        self.add_flow(segpostprocessor_op, visualizer_sink, {("", "receivers")})
        #>>>>>>>>>>>>>>>>>


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
    cmdline_args = parser.parse_args()

    if cmdline_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = WebRTCClientApp(cmdline_args)

    with Tracker(app, filename=cmdline_args.logger_filename) as tracker:
        app.run()
