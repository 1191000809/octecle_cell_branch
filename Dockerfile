FROM python:3.9-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# ### for cv v1
USER root
RUN apt-get update && apt-get install sudo
USER user

COPY ./ /opt/app/

USER root
# ADD /opt/app/source.list /etc/apt
## 换源
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list  
RUN cat /etc/apt/sources.list
RUN apt-get clean
##
RUN sudo apt update --fix-missing
# RUN sudo apt-get upgrade
RUN sudo apt install -y libgl1-mesa-glx 
# -i http://pypi.douban.com/simple/
USER user

USER root
RUN sudo apt update 
# RUN sudo apt install -y libcairo2
RUN sudo apt install -y libglib2.0-dev
USER user


RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
