from loguru import logger
import os
import subprocess

from .auto import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .device import get_device, get_torch_device_module

# Remove the original logger in Python to prevent duplicate printing.
logger.remove()

_LOGGER = logger


def get_logger():
    return _LOGGER


def get_repo_git_info(repo_path):
    original_directory = os.getcwd()
    os.chdir(repo_path)

    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')

        commit_id = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')

        remote_url = subprocess.check_output(
            ['git', 'remote', 'get-url', 'origin'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')

        return branch, commit_id, remote_url
    except subprocess.CalledProcessError as e:
        return None, None, None
    finally:
        os.chdir(original_directory)


__all__ = [
    'AutoConfig', 'AutoModelForCausalLM', 'AutoTokenizer', 'get_device',
    'get_torch_device_module'
]
