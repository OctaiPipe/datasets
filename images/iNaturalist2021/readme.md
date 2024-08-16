# iNaturalist dataset
This is a picture dataset of animals and plants fround here: https://github.com/visipedia/inat_comp/tree/master/2021
It was used in this paper for [FL Federated Visual Classification with Real-World Data Distribution](https://arxiv.org/pdf/2003.08082)because it can naturally be partitionned by user, and makes natural non-IID data.

## Information about the dataset
The training set contains 2,686,843 images, from 158,873 users.

There are 10,000 species, but thanks to phylogeny (the hierarchical organisation of all living organisms), we can naturally group categories together. Yes: Biology is amazing!  
So there are 13 phylums, 51 classes, 1103 families, which are likely to be the most useful to us. This allows us to reduce the number of categories without having to exclude any images.

## Metadata available
The raw metadata is stored in the json files, and has been processed into several csv files for our own convenience. All files can be found in the blob storage (see below how to access)
From the original dataset, I have renamed several columns, for our ease of understanding. I renamed "name" as "species", to make it clear the name is the species name, and "rights_holder" as "user". 

The table in training_image_info.csv contains one row per image, and contains the following columns: image_id, user, width, height, file_name, phylum, class, order, species
It has a shape of 2686843 rows × 11 columns

The table in images_and_categories_per_users.csv contains one row per users, and the aggregation of: the total number of images the user has taken, and the distinct number of phylums, classes and orders and species.
Its size is 158873 rows × 6 columns

### How to access the metadata
These two tables have been stored in the blob storage, and can be downloaded as such in Python: 
```python
from azure.storage.blob import BlobServiceClient

# Replace these values with your information
connection_string = "BlobEndpoint=https://octaipipedatasets.blob.core.windows.net/;QueueEndpoint=https://octaipipedatasets.queue.core.windows.net/;FileEndpoint=https://octaipipedatasets.file.core.windows.net/;TableEndpoint=https://octaipipedatasets.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=b&srt=sco&sp=rl&se=2025-09-01T22:55:57Z&st=2024-08-16T14:55:57Z&spr=https&sig=bVnZQ31gYb13mqvaEvWAchc4qwzgR77zwJmWIkp8Uy0%3D"
container_name = "inaturalist"
blob_name = "2021/training_images_and_categories_per_users.csv"
download_file_path = "training_images_and_categories_per_users.csv"

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get the container client
container_client = blob_service_client.get_container_client(container_name)

# Get the blob client
blob_client = container_client.get_blob_client(blob_name)

try:
    # Download the blob to a local file
    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    print(f"Downloaded blob {blob_name} to {download_file_path}")

except Exception as e:
    print(f"An error occurred: {e}")
```





