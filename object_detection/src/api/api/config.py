import os

# general
# endpoint configuration
MAX_ALLOWED_LIMIT = 200

# URLs and connection setting to other system applications (message broker, database, shared volume)
RABBITMQ_BROKER_URL = os.environ["RABBITMQ_BROKER_URL"]
RABBITMQ_QUEUES = {
    "inference": os.environ["RABBITMQ_INFERENCE_QUEUE"],
}
MAX_QUEUE_LENGTH = int(os.environ["RABBITMQ_MAX_QUEUE_LENGTH"])
WORKER_TIMEOUT_SECONDS = 15 * 60
POSTGRES_URL = os.environ["POSTGRES_URL"]
