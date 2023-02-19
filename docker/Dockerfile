FROM nvcr.io/nvidia/pytorch:22.12-py3
WORKDIR /workspace
EXPOSE 5000

RUN apt update
RUN apt install -y libgl1-mesa-dev git libglib2.0-0 g++ gcc
RUN pip install Flask flask_cors opencv-python scikit-learn tqdm scipy scikit-image autopep8 grad-cam
