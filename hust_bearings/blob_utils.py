import logging
import os

from azure.storage.blob import BlobServiceClient


def get_container_client(connection_string: str,
                         container: str):
    """Get Azure container client

    Args:
        connection_string (str): connection string to the blob storage
                                 account
        container (str): name of the container
    Returns:
        container_client: Azure container client
    """

    blob_service_client = BlobServiceClient \
        .from_connection_string(connection_string)
    container_client = blob_service_client \
        .get_container_client(container)

    return container_client


def download_blob(
        connection_string: str,
        container: str = 'cwru-bearings',
        blob_name: str = None,
        work_dir: str = './',
        destination_dir: str = 'data_new',):
    """Download blob from Azure container

    Args:
        connection_string (str): connection string to the blob storage
                                 account
        container (str, optional): name of the container.
            Defaults to 'cwru-bearings'
        blob_name (str, optional): name of the blob. Defaults to None
        work_dir (str, optional): local work dir. Defaults to './'
        destination_dir (str, optional): Defaults to 'chat_history'

    Returns:
        output_path (str)
    """
    logger = logging.getLogger(__name__)
    dir_path = os.path.join(work_dir, destination_dir)
    os.makedirs(dir_path, exist_ok=True)
    output_path = os.path.join(dir_path, blob_name)
    logger.info('Initialising Azure blob container client')
    container_client = get_container_client(
        connection_string=connection_string,
        container=container)

    logger.info(f'Downloading blob {blob_name} from container {container} \
to {output_path}')
    try:
        blob_data = container_client.download_blob(blob_name)
    except Exception as e:
        logger.error('Unable to download blob')
        logger.error(e)
        blob_data = None
        return None
    with open(output_path, "wb") as my_blob:
        blob_data.readinto(my_blob)
        return output_path


def download_container(
        connection_string: str,
        container: str = 'cwru-bearings',
        work_dir: str = './',
        destination_data_dir: str = 'data',
        dataset_subdir: str = 'data',):
    """Download all blobs from an Azure container to a specified local folder

    Args:
        connection_string (str): connection string to the blob storage
                                 account
        container (str, optional): name of the container.
            Defaults to 'cwru-bearings'
        blob_name (str, optional): name of the blob. Defaults to None
        work_dir (str, optional): local work dir. Defaults to './'
        destination_data_dir (str, optional): Defaults to 'data'
        dataset_subdir (str, optional): sub-dir destination_data_dir to save
            files to.
            Defaults to 'knowledge_base'

    Returns:
        output_dir (str)
    """
    logger = logging.getLogger(__name__)
    output_dir = os.path.join(work_dir,
                              destination_data_dir, dataset_subdir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info('Initialising Azure blob container client')
    container_client = get_container_client(
        connection_string=connection_string,
        container=container)

    logger.info(f'Downloading files in container {container} to {output_dir}')
    for blob_ in container_client.walk_blobs():
        blob_name = blob_.name
        with open(os.path.join(output_dir, blob_name), "wb") as my_blob:
            blob_data = container_client.download_blob(blob_name)
            blob_data.readinto(my_blob)

    return output_dir


def save_file_to_blob(
        connection_string: str,
        filepath: str,
        container: str = 'cwru-bearings',
        blob_name: str = None,
        container_create_specs: dict = {}):
    """Upload a file to an Azure blob

    Args:
        connection_string (str): connection string to the blob storage
                                 account
        filepath (str)
        container (str, optional): name of the container.
            Defaults to 'cwru-bearings'
        blob_name (str, optional): name of the blob. Defaults to None
        container_create_specs (dict, optional):
            specs for creating the container if not exist.
            Defaults to {}

    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(filepath):
        logger.error(f'{filepath} does not exist')
        return None

    logger.info('Initialising Azure blob service client')
    blob_service_client = BlobServiceClient \
        .from_connection_string(connection_string)

    container_names = []
    for container_ in blob_service_client.list_containers():
        container_names.append(container_.name)
    if container not in container_names:
        logger.warning(f'container {container} does not exist; create one')
        blob_service_client.create_container(name=container,
                                             **container_create_specs)
    container_client = blob_service_client \
        .get_container_client(container)

    logger.info(f'Uploading file to container {container}')
    with open(filepath, "rb") as data:
        container_client.upload_blob(blob_name, data,
                                     blob_type="BlockBlob",
                                     overwrite=True)

    return True


def save_folder_to_container(
        connection_string: str,
        dir_path: str,
        container: str = 'cwru-bearings',
        container_create_specs: dict = {}):
    """Upload a folder to an Azure container

    Args:
        connection_string (str): connection string to the blob storage
                                 account
        dir_path (str)
        container (str, optional): name of the container.
            Defaults to 'cwru-bearings'
        container_create_specs (dict, optional):
            specs for creating the container if not exist.
            Defaults to {}

    """
    logger = logging.getLogger(__name__)

    logger.info('Initialising Azure blob service client')
    blob_service_client = BlobServiceClient \
        .from_connection_string(connection_string)

    container_names = []
    for container_ in blob_service_client.list_containers():
        container_names.append(container_.name)
    if container not in container_names:
        logger.warning(f'container {container} does not exist; create one')
        blob_service_client.create_container(name=container,
                                             **container_create_specs)
    container_client = blob_service_client \
        .get_container_client(container)

    logger.info(f'Uploading files to container {container}')
    if not os.path.exists(dir_path):
        return None
    for filename_ in os.listdir(dir_path):
        filepath_ = os.path.join(dir_path, filename_)
        blob_name = filename_
        with open(filepath_, "rb") as data:
            container_client.upload_blob(blob_name, data,
                                         blob_type="BlockBlob",
                                         overwrite=True)

    return True


def delete_blob(connection_string: str,
                container: str = 'cwru-bearings',
                blob_name: str = None,):
    """Delete a blob on Azure

    Args:
        connection_string (str): connection string to the blob storage
                                 account
        blob_name (str, optional): name of the blob.
            Defaults to None

    """
    logger = logging.getLogger(__name__)
    logger.info(f'Initialising Azure blob container client {container}')
    container_client = get_container_client(
        connection_string=connection_string,
        container=container)

    logger.warning(f'Deleting blob {blob_name}')
    try:
        container_client.delete_blob(blob=blob_name)
        return True
    except Exception as e:
        logger.error('Failed to delete blob')
        logger.error(e)
        return False
