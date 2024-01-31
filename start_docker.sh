WS=/root/sp_ws
DOCKERIMAGE="sp_image:pc"
NN_MODELS="/home/dji/workspace/D2SLAM-Fusion/tools-NNModels-generator"
script_path=$(dirname "$(readlink -f "$0")")
CURRENT_DIR=$script_path

docker run -it --rm --runtime=nvidia --gpus all  --net=host \
    -v ${CURRENT_DIR}:${WS}/sp_ws \
    -v ${NN_MODELS}:${WS}/NNmodels_generator \
    -v /dev/:/dev/  --privileged --env="DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix  --name="sp_container" ${DOCKERIMAGE} /bin/bash 