# v4l

## Useful commands

* sudo dmesg

Connect use camera and run

```bash
sudo dmesg | grep -i logitech
[ 2643.951270] usb 3-1.1: Product: Logitech Webcam C930e
[ 2643.973344] uvcvideo 3-1.1:1.0: Found UVC 1.00 device Logitech Webcam C930e (046d:0843)
[ 2730.030694] usb 3-2.1: Product: Logitech Webcam C930e
[ 2730.032029] uvcvideo 3-2.1:1.0: Found UVC 1.00 device Logitech Webcam C930e (046d:0843)
[ 3146.027900] usb 3-2.1: Product: Logitech Webcam C930e
[ 3146.029170] uvcvideo 3-2.1:1.0: Found UVC 1.00 device Logitech Webcam C930e (046d:0843)
[ 3316.587124] usb 3-1.1: Product: Logitech Webcam C930e
[ 3316.588598] uvcvideo 3-1.1:1.0: Found UVC 1.00 device Logitech Webcam C930e (046d:0843)
```

* lusb
```bash 
lsusb | grep -i logitech
Bus 003 Device 015: ID 046d:0843 Logitech, Inc. Webcam C930e
```
```bash
lsusb -v -d 046d:0843
```


* List devices
```bash
v4l2-ctl --list-devices


Logitech Webcam C930e (usb-0000:00:14.0-1.1):
	/dev/video48
	/dev/video49
	/dev/media1

```


## Debugging features of `v4l2` device

* /dev/video0
```bash
$v4l2-ctl -d /dev/video0 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 1280x720
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x180
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
```

* /dev/video4

```bash
$v4l2-ctl -d /dev/video4 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)

...

		Size: Discrete 2304x1536
			Interval: Discrete 0.500s (2.000 fps)
	[1]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)

		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)

```


* USB endoscope camera (1/9 inch sensor size; 30fps; 70CAngleView)
```bash
v4l2-ctl -d /dev/video4 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 352x288
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 176x144
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)

```


* logitech

/dev/video48
```bash

v4l2-ctl -d /dev/video48 --list-formats-ext


v4l2-ctl -d /dev/video48 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)


...


		Size: Discrete 1920x1080
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 2304x1296
			Interval: Discrete 0.500s (2.000 fps)
		Size: Discrete 2304x1536
			Interval: Discrete 0.500s (2.000 fps)
	[1]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)

...

		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)


```

