FROM nvcr.io/nvidia/tensorrt:23.02-py3

ARG OPENCV_VERSION=4.7.0

RUN  apt-get -y update && \
      apt-get install -y wget curl lsb-release git \
      libatlas-base-dev \
      libeigen3-dev \
      libgoogle-glog-dev \
      libsuitesparse-dev \
      libglib2.0-dev \
      libyaml-cpp-dev \
      libdw-dev \
      vim

RUN pip install onnxruntime-gpu tqdm 

#Install ROS
# update ros repository
# some code copied from https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/docker/Dockerfile
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
# # RUN sh -c 'echo "deb http://mirrors.ustc.edu.cn/ros/ubuntu/ `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' && \
#       curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
#       apt-get update && \
#       if [ "x$(nproc)" = "x1" ] ; then export USE_PROC=1 ; \
#       else export USE_PROC=$(($(nproc)/2)) ; fi && \
#       apt-get update && \
#       DEBIAN_FRONTEND=noninteractive  apt-get install -y \
#       ros-${ROS_VERSION}-ros-base \
#       ros-${ROS_VERSION}-nav-msgs \
#       ros-${ROS_VERSION}-sensor-msgs \
#       ros-${ROS_VERSION}-cv-bridge \
#       ros-${ROS_VERSION}-rviz \
#       ros-${ROS_VERSION}-pcl-ros \
#       ros-${ROS_VERSION}-image-transport-plugins \
#       python3-rosdep \
#       python3-rosinstall \
#       python3-rosinstall-generator \
#       python3-wstool \
#       build-essential \
#       python3-rosdep \
#       ros-${ROS_VERSION}-catkin \
#       net-tools \
#       python3-catkin-tools \
#       htop \
#       xterm \
#       gdb && \
#       rosdep init && \
#       rosdep update

#Install OpenCV4 with CUDA
RUN   apt update && \
      apt install libgtk2.0-dev -y && \
      wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip && \
      unzip opencv.zip && \
      rm opencv.zip && \
      git clone https://github.com/opencv/opencv_contrib.git -b ${OPENCV_VERSION}
RUN   cd opencv-${OPENCV_VERSION} && \
      mkdir build && cd build && \
      cmake .. \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=/usr/local \
            -D WITH_CUDA=ON \
            -D WITH_CUDNN=ON \
            -D WITH_CUBLAS=ON \
            -D WITH_TBB=ON \
            -D OPENCV_DNN_CUDA=ON \
            -D OPENCV_ENABLE_NONFREE=ON \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
            -D BUILD_EXAMPLES=OFF \
            -D BUILD_opencv_java=OFF \
            -D BUILD_opencv_python=OFF \
            -D BUILD_TESTS=OFF \
            -D BUILD_PERF_TESTS=OFF \
            -D BUILD_opencv_apps=OFF \
            -D BUILD_LIST=calib3d,features2d,highgui,dnn,imgproc,imgcodecs,\
cudev,cudaoptflow,cudaimgproc,cudalegacy,cudaarithm,cudacodec,cudastereo,\
cudafeatures2d,xfeatures2d,tracking,stereo,\
aruco,videoio,ccalib && \
      make -j$(nproc) && \
      make install 

WORKDIR /root/sp_ws