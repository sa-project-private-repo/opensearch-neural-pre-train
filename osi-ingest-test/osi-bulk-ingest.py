#!/usr/bin/env python3
"""
OSI Bulk Ingestion Script with Connection Pooling.

Usage:
    ./osi-bulk-ingest.py --count 1000 --batch-size 100 --workers 4
"""

import argparse
import random
import string
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Generator

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OSI_ENDPOINT = "https://osi-vpc-internal-connection-6scbjbll7ezxlhrbigvyzx3squ.us-east-1.osis.amazonaws.com"
OSI_PATH = "/http-094569/logs"
REGION = "us-east-1"


@dataclass
class IngestStats:
    """Statistics for ingestion."""

    total_docs: int = 0
    success_docs: int = 0
    failed_docs: int = 0
    total_batches: int = 0
    success_batches: int = 0
    failed_batches: int = 0
    elapsed_time: float = 0.0

    @property
    def docs_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.success_docs / self.elapsed_time


class OSIClient:
    """OSI client with connection pooling and AWS SigV4 authentication."""

    _instance: "OSIClient | None" = None
    _session: requests.Session | None = None

    def __new__(cls) -> "OSIClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_session()
        return cls._instance

    def _init_session(self) -> None:
        """Initialize session with connection pooling."""
        self._session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=100,
            max_retries=retry_strategy,
        )

        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self._credentials = boto3.Session().get_credentials()

    def _sign_request(self, url: str, data: str) -> dict[str, str]:
        """Sign request with AWS SigV4."""
        request = AWSRequest(method="POST", url=url, data=data)
        request.headers["Content-Type"] = "application/json"
        SigV4Auth(self._credentials, "osis", REGION).add_auth(request)
        return dict(request.headers)

    def send_batch(self, records: list[dict]) -> tuple[bool, str]:
        """Send a batch of records to OSI."""
        import json

        url = f"{OSI_ENDPOINT}{OSI_PATH}"
        data = json.dumps(records)
        headers = self._sign_request(url, data)

        try:
            response = self._session.post(url, data=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return True, "OK"
            return False, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return False, str(e)


class RandomDataGenerator:
    """Generate random log data."""

    LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    SOURCES = ["api-gateway", "auth-service", "user-service", "payment-service", "order-service"]
    ACTIONS = ["login", "logout", "create", "update", "delete", "read", "search", "export"]
    STATUSES = ["success", "failure", "pending", "timeout", "cancelled"]

    @classmethod
    def generate_record(cls) -> dict:
        """Generate a single random log record."""
        timestamp = datetime.now(UTC) - timedelta(
            seconds=random.randint(0, 86400)
        )

        return {
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": random.choice(cls.LOG_LEVELS),
            "source": random.choice(cls.SOURCES),
            "action": random.choice(cls.ACTIONS),
            "status": random.choice(cls.STATUSES),
            "user_id": f"user-{random.randint(1, 10000)}",
            "request_id": str(uuid.uuid4()),
            "duration_ms": random.randint(1, 5000),
            "message": cls._generate_message(),
            "metadata": {
                "ip": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                "region": random.choice(["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-2"]),
                "version": f"{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,99)}",
            },
        }

    @classmethod
    def _generate_message(cls) -> str:
        """Generate a random log message."""
        templates = [
            "Request processed successfully",
            "User authentication completed",
            "Database query executed in {ms}ms",
            "Cache hit for key: {key}",
            "Connection timeout after {ms}ms",
            "Rate limit exceeded for user {user}",
            "Invalid request parameters: {param}",
            "Service health check passed",
            "Background job completed: {job}",
            "File uploaded: {file}",
        ]
        msg = random.choice(templates)
        return msg.format(
            ms=random.randint(1, 1000),
            key="".join(random.choices(string.ascii_lowercase, k=8)),
            user=f"user-{random.randint(1, 1000)}",
            param=random.choice(["id", "email", "token", "page"]),
            job=random.choice(["cleanup", "sync", "report", "backup"]),
            file=f"file-{random.randint(1, 100)}.json",
        )

    @classmethod
    def generate_batch(cls, size: int) -> list[dict]:
        """Generate a batch of random records."""
        return [cls.generate_record() for _ in range(size)]


def ingest_batch(batch_id: int, records: list[dict]) -> tuple[int, bool, str]:
    """Ingest a single batch (worker function)."""
    client = OSIClient()
    success, message = client.send_batch(records)
    return batch_id, success, message


def run_bulk_ingest(
    total_count: int,
    batch_size: int,
    workers: int,
) -> IngestStats:
    """Run bulk ingestion with connection pooling."""
    stats = IngestStats()
    stats.total_docs = total_count

    # Pre-initialize client (singleton)
    _ = OSIClient()

    # Generate all batches
    batches: list[tuple[int, list[dict]]] = []
    for i in range(0, total_count, batch_size):
        size = min(batch_size, total_count - i)
        batches.append((len(batches), RandomDataGenerator.generate_batch(size)))

    stats.total_batches = len(batches)
    print(f"Generated {stats.total_batches} batches, starting ingestion...")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(ingest_batch, batch_id, records): batch_id
            for batch_id, records in batches
        }

        for future in as_completed(futures):
            batch_id, success, message = future.result()
            batch_size_actual = len(batches[batch_id][1])

            if success:
                stats.success_batches += 1
                stats.success_docs += batch_size_actual
            else:
                stats.failed_batches += 1
                stats.failed_docs += batch_size_actual
                print(f"  Batch {batch_id} failed: {message}")

            # Progress
            completed = stats.success_batches + stats.failed_batches
            if completed % 10 == 0 or completed == stats.total_batches:
                print(f"  Progress: {completed}/{stats.total_batches} batches")

    stats.elapsed_time = time.time() - start_time
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="OSI Bulk Ingestion with Connection Pooling")
    parser.add_argument("--count", type=int, default=1000, help="Total documents to ingest")
    parser.add_argument("--batch-size", type=int, default=100, help="Documents per batch")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    args = parser.parse_args()

    print(f"OSI Bulk Ingestion")
    print(f"  Total docs: {args.count}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.workers}")
    print()

    stats = run_bulk_ingest(args.count, args.batch_size, args.workers)

    print()
    print("Results:")
    print(f"  Success: {stats.success_docs}/{stats.total_docs} docs")
    print(f"  Batches: {stats.success_batches}/{stats.total_batches} succeeded")
    print(f"  Time: {stats.elapsed_time:.2f}s")
    print(f"  Throughput: {stats.docs_per_second:.1f} docs/sec")


if __name__ == "__main__":
    main()
