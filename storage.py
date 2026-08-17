import os
import logging

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_ENDPOINT'),
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
        config=Config(signature_version='s3v4'),
        region_name='us-east-1',
    )


def download_file(object_name):
    """
    Fetch a document previously uploaded by the source-predicate crawler, by
    its bare object key (e.g. "united_kingdom/1/172_doc.pdf") — never a
    "s3://bucket/key" URI; the bucket name lives in MINIO_BUCKET, not the
    stored key (see that repo's app/storage.py::upload_file for why).
    Returns the raw bytes.
    """
    s3 = get_s3_client()
    bucket_name = os.getenv('MINIO_BUCKET')
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=object_name)
        return obj['Body'].read()
    except Exception as e:
        logger.error(f"Failed to download file from S3 ({object_name}): {e}")
        raise
