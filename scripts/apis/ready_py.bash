
SOURCE=$1 #replayer #v4l2
cd /workspace/volumes/ready/src/ready/apis/holoscan/ready/python

#TODO create a config file to comment/uncomment models paths and names
#clear && python ready.py -c ready-mobious.yaml -d /workspace/volumes/datasets/ready/mobious/models -m _weights_10-09-24_06-35-14-sim-BHWC.onnx -l logger.log -df TRUE -s replayer #v4l2
#clear && python ready.py -c ready-mobious.yaml -d /workspace/volumes/datasets/ready/mobious/models -m _weights_14-12-24_19-25-26-sim-BHWC.onnx -l logger.log -df TRUE -s replayer #v4l2
#clear && python ready.py -c ready-mobious.yaml -vp /workspace/volumes/datasets/ready/videos/novel -d /workspace/volumes/datasets/ready/mobious/models -m _weights_15-12-24_07-00-10-sim-BHWC.onnx -l logger.log -df TRUE -s replayer #v4l2

clear && python ready.py -c ready-mobious.yaml -l logger.log -df TRUE -s ${SOURCE}
