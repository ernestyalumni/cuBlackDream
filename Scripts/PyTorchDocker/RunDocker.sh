#!/bin/bash
# An example for the user input would be
# /home/propdev/Prop/cuBlackDream:/cuBlackDream

# Check if there is any user input
if [ -z "$1" ]; then

  # No user input, so prompt for an absolute path
	echo "Enter an absolute path"
  read path_to_mount

else

  # User input provided, so that the first argument as path
  path_to_mount="$1"
fi

# Check if path is an existing path
# [ is start of a test command condition. -d is a test for a directory.
if [ ! -d "$path_to_mount" ]; then
  echo "The path '$path_to_mount' is not an existing path."
  exit 1
fi

# Run command
command="docker run -v $path_to_mount:/cuBlackDream --gpus all -it -p "
command+="8888:8888 --rm --ipc=host --ulimit memlock=-1 --ulimit "
command+="stack=67108864 nvcr.io/nvidia/pytorch:23.08-py3 "

echo $command
# eval is part of POSIX and executes arguments like a shell command.
eval $command