# Glass to glass latency


## webrtc_client for ready api
We are experimenting with different values of branch_hz (10, 15, and 20) with [webrtc_client.py](../../../src/ready/apis/holoscan/webrtc_ready/webrtc_client.py), capturing screenshots at the beginning, middle, and end of a one-minute stream. Notably, for branch_hz = 15, the glass-to-glass latency is a bit inconsistent, with observed values of 344ms at ~20s, 286ms at ~30s, and 229ms at ~57s. See below other values of latency for branch_hz=10 and branch_hz=20, where the latter is definitely increasing glass to glass latency.

![fig](figure.svg)
