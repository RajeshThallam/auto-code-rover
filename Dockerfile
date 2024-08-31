# for building auto-code-rover:latest
FROM yuntongzhang/swe-bench:latest

RUN git config --global user.email acr@nus.edu.sg
RUN git config --global user.name acr

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y vim build-essential libssl-dev

RUN wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-490.0.0-linux-x86_64.tar.gz \
-O /tmp/google-cloud-sdk.tar.gz | bash

RUN mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xvzf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh -q

ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

COPY . /opt/auto-code-rover

WORKDIR /opt/auto-code-rover/demo_vis/front
RUN sed -i 's/\r$//' /opt/auto-code-rover/demo_vis/run.sh
RUN apt install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get install nodejs -y
RUN npm i
RUN npm run build

WORKDIR /opt/auto-code-rover
RUN conda env create -f environment.yml

EXPOSE 3000 5000
ENTRYPOINT [ "/bin/bash" ]
