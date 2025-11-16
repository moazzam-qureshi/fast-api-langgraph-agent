"""Celery configuration for background task processing."""
from celery import Celery
import os

# Create Celery instance
celery = Celery(
    'fastapi_rag',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    include=['services.ingestion']  # Include task modules
)

# Configure Celery
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes hard limit
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Retry configuration
celery.conf.task_default_retry_delay = 60  # 1 minute
celery.conf.task_max_retries = 3

if __name__ == '__main__':
    celery.start()