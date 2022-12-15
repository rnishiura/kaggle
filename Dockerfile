FROM kaggle/python:latest

RUN apt-get update
RUN apt-get install -y gcc g++ locales git procps vim tmux
RUN locale-gen ja_JP.UTF-8
RUN localedef -f UTF-8 -i ja_JP ja_JP
ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

WORKDIR /kaggle

