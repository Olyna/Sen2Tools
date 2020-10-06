#
# TO USE:
#
# sudo docker pull ubuntu
#
# sudo docker build . -t sen2tools
#
# sudo docker run -it --name devtest --mount type=bind,source="$(pwd)",target=/UNet_Docker sen2tools
#


FROM ubuntu

COPY . /UNet_Docker
WORKDIR /UNet_Docker

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y python3-pip

RUN apt-get install -y python3-matplotlib
RUN pip3 install -r requirements.txt

RUN apt-get install -y git
RUN git clone https://github.com/Olyna/Sen2Tools.git
RUN cd Sen2Tools/
RUN pip3 install .

RUN pwd

#RUN python3 ./lissage_v01.2.py