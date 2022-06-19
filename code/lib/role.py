#!/usr/bin/env python
import json
import time
from dataclasses import dataclass, field
from typing import Dict

import boto3

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

    _logger = logger
    _iam_client = boto3.client("iam")

    @classmethod
    def _retrieve_role(cls, role_config: SagemakerRoleConfig) -> SagemakerRole:
        """Retrieve role details from role configuration

        Args:
            role_config (SagemakerRoleConfig): sagemaker role configuration data

        Returns:
            SagemakerRole: sagemaker role data

        Raises:
            NoSuchEntityException: role does not exists
        """

        cls._logger.info(f"Retrieving role: {role_config.name}")
        role_arn = cls._iam_client.get_role(RoleName=role_config.name)["Role"]["Arn"]
        cls._logger.info("Role retrieved successfully")
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
        """

        cls._logger.info(f"Creating role: {role_config.name}")

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

        # TODO: Add more robust way of checking the policy creation
        # wait untill role is created
        time.sleep(5)
        cls._logger.info("Role created successfully.")
        return SagemakerRole(arn=role_arn, config=role_config)

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
            cls._logger.info("Role does not exists")

        try:
            return cls._create_role(role_config)
        except cls._iam_client.exceptions.EntityAlreadyExistsException:
            cls._logger.error("Role already exists")
            raise RoleHandlerException(
                role_config=role_config,
                message=f"Unexpected error during initialization of the role: {role_config}",
            )
