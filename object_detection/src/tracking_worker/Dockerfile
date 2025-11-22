FROM ultralytics/ultralytics:8.2.50-jetson-jetpack5

COPY ./src/tracking_worker /tracking_worker
COPY ./src/commonlib /commonlib

# install python dependencies
ENV APP_DIR='/tracking_worker'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# install custom package
WORKDIR /commonlib
RUN python3 -m build && \
    pip install $(ls -1 dist/commonlib*.tar.gz | tail -1)

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# run application
WORKDIR $APP_DIR
CMD ["python3", "/tracking_worker/worker.py"]
