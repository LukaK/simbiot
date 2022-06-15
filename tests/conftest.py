#!/usr/bin/env python
import os
import boto3
import pytest
from moto import mock_iam


@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture
def iam_client(aws_credentials):
    with mock_iam():
        iam_client = boto3.client("iam")
        yield iam_client
