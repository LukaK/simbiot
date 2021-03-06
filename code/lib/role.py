#!/usr/bin/env python
import json
from dataclasses import dataclass, field
from typing import Dict

import boto3
from retry import retry

from .logger import logger


@dataclass
class SagemakerRoleConfig:
    name: str
    policy_arn: str = field(
        default="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess", init=False
    )
    trust_policy_document: Dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.trust_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }


@dataclass
class SagemakerRole:
    arn: str
    name: str = field(init=False)
    config: SagemakerRoleConfig = field(repr=False)

    def __post_init__(self):
        self.name = self.config.name


class RoleHandlerException(Exception):
    def __init__(self, role_config: SagemakerRoleConfig, message: str):
        self.role_config = role_config
        Exception.__init__(self, message)


class RoleHandler:

    _iam_client = boto3.client("iam")

    @classmethod
    @retry(_iam_client.exceptions.NoSuchEntityException, delay=5, tries=2)
    def _retrieve_role(cls, role_config: SagemakerRoleConfig) -> SagemakerRole:
        """Retrieve role details from role configuration

        Args:
            role_config (SagemakerRoleConfig): sagemaker role configuration data

        Returns:
            SagemakerRole: sagemaker role data

        Raises:
            NoSuchEntityException: role does not exists
        """

        logger.info(f"Retrieving role: {role_config.name}")
        role_arn = cls._iam_client.get_role(RoleName=role_config.name)["Role"]["Arn"]
        logger.info("Role retrieved successfully")
        return SagemakerRole(arn=role_arn, config=role_config)

    @classmethod
    def _create_role(cls, role_config: SagemakerRoleConfig) -> SagemakerRole:
        """Create the role.

        Args:
            role_config (SagemakerRoleConfig): role configuration data

        Returns:
            SagemakerRole: role information data

        Raises:
            EntityAlreadyExistsException: role allready exists
            NoSuchEntityException: error during retrieving role
        """

        logger.info(f"Creating role: {role_config.name}")

        # create role
        role_arn = cls._iam_client.create_role(
            RoleName=role_config.name,
            AssumeRolePolicyDocument=json.dumps(role_config.trust_policy_document),
        )["Role"]["Arn"]

        # attach policy
        cls._iam_client.attach_role_policy(
            RoleName=role_config.name,
            PolicyArn=role_config.policy_arn,
        )

        return cls._retrieve_role(role_config)

    @classmethod
    def initialize_role(cls, role_config: SagemakerRoleConfig) -> SagemakerRole:
        """Initialize/setup permissions

        Args:
            role_config (SagemakerRoleConfig): role configuration data

        Returns:
            SagemakerRole: role data
        """

        try:
            return cls._retrieve_role(role_config)
        except cls._iam_client.exceptions.NoSuchEntityException:
            logger.info("Role does not exists")

        try:
            return cls._create_role(role_config)
        except (
            cls._iam_client.exceptions.EntityAlreadyExistsException,
            cls._iam_client.exceptions.NoSuchEntityException,
        ):
            logger.error("Role already exists")
            raise RoleHandlerException(
                role_config=role_config,
                message=f"Unexpected error during initialization of the role: {role_config}",
            )
