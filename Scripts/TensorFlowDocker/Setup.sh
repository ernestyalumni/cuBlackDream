# Reference https://www.tensorflow.org/install/docker "Download a TensorFlow Docker image"
# Official TensorFlow Docker images located in tensorflow/tensorflow.
# Tag latest - latest release of TensorFlow CPU binary image, Default.
# Each base tag has variants: tag-gpu.

docker pull tensorflow/tensorflow:latest-gpu-jupyter # latest release with GPU support and Jupyter

# Create a Docker group if you haven't already.
sudo groupadd docker

# Check who is a member of these groups
grep /etc/group -e "docker"
grep /etc/group -e "sudo"

# Add your user to this docker group:

echo $USER
sudo usermod -aG docker

docker run --gpus all -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter

# The following worked, but did not bind the mount to be seen on jupyter notebooks.
# docker run -v /home/propdev/Prop/cuBlackDream:/cuBlackDream --gpus all -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter

# -p 8888:8888 maps port 8888 inside container to port 8888 on host machine.

docker run -v /home/propdev/Prop/cuBlackDream:/cuBlackDream --gpus all -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter jupyter notebook --notebook-dir=/

docker run -v /home/propdev/Prop/cuBlackDream:/cuBlackDream --gpus all -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter jupyter notebook --notebook-dir=/cuBlackDream --ip 0.0.0.0 --allow-root

# Pytorch
# Use the NGC (Nvidia GPU Cloud) (Docker) container


