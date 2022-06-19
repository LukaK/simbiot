#!/usr/bin/env python
import pytest
from moto.core import patch_client


def test__sagemaker_role_config_initialization():
    from lib.role import SagemakerRoleConfig

    role_config = SagemakerRoleConfig(name="test_role")


def test__role_handler_exception_initialization():
    from lib.role import RoleHandlerException, SagemakerRoleConfig

    with pytest.raises(RoleHandlerException):
        role_config = SagemakerRoleConfig(name="test_role")
        raise RoleHandlerException(role_config=role_config, message="Test message")


def test__sagemaker_role_initialization():
    from lib.role import SagemakerRole, SagemakerRoleConfig

    role_config = SagemakerRoleConfig(name="test_role")
    role_config = SagemakerRole(arn="test_arn", config=role_config)


def test__role_handler_retrieve_role_not_exists(iam_client):
    from lib.role import RoleHandler, SagemakerRoleConfig

    patch_client(RoleHandler._iam_client)

    role_handler = RoleHandler()
    role_config = SagemakerRoleConfig(name="test_role")
    with pytest.raises(role_handler._iam_client.exceptions.NoSuchEntityException):
        role_handler._retrieve_role(role_config)


@pytest.mark.slow
def test__role_handler_retrieve_role_exists(iam_client):
    from lib.role import RoleHandler, SagemakerRoleConfig

    patch_client(RoleHandler._iam_client)

    role_handler = RoleHandler()
    role_config = SagemakerRoleConfig(name="test_role")
    role_handler._create_role(role_config)
    role_handler._retrieve_role(role_config)


@pytest.mark.slow
def test__role_handler_create_role_not_exists(iam_client):
    from lib.role import RoleHandler, SagemakerRoleConfig

    patch_client(RoleHandler._iam_client)

    role_handler = RoleHandler()
    role_config = SagemakerRoleConfig(name="test_role")
    role_handler._create_role(role_config)


@pytest.mark.slow
def test__role_handler_create_role_exists(iam_client):
    from lib.role import RoleHandler, SagemakerRoleConfig

    patch_client(RoleHandler._iam_client)

    role_handler = RoleHandler()
    role_config = SagemakerRoleConfig(name="test_role")
    role_handler._create_role(role_config)
    with pytest.raises(RoleHandler._iam_client.exceptions.EntityAlreadyExistsException):
        role_handler._create_role(role_config)


@pytest.mark.slow
def test__role_handler_initialize_role_not_exists(iam_client):
    from lib.role import RoleHandler, SagemakerRole, SagemakerRoleConfig

    patch_client(RoleHandler._iam_client)

    role_handler = RoleHandler()
    role_config = SagemakerRoleConfig(name="test_role")
    role = role_handler.initialize_role(role_config)
    assert isinstance(role, SagemakerRole)


@pytest.mark.slow
def test__role_handler_initialize_role_exists(iam_client):
    from lib.role import RoleHandler, SagemakerRole, SagemakerRoleConfig

    patch_client(RoleHandler._iam_client)

    role_handler = RoleHandler()
    role_config = SagemakerRoleConfig(name="test_role")
    role_handler._create_role(role_config)
    role = role_handler.initialize_role(role_config)
    assert isinstance(role, SagemakerRole)
