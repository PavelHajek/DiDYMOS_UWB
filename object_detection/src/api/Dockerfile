FROM python:3.10.14-slim

COPY ./src/api /api
COPY ./src/commonlib /commonlib

# install python dependencies
ENV APP_DIR='/api'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# install custom package
WORKDIR /commonlib
RUN python -m build && \
    pip install $(ls -1 dist/commonlib*.tar.gz | tail -1)

# expose ports and run application
WORKDIR $APP_DIR
EXPOSE $APP_PORT
CMD uvicorn rest_server:app \
    --host $API_HOST \
    --port $API_PORT \
    --log-config ./logging.yaml \
    --log-level info
