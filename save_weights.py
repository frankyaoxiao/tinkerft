import tinker
import urllib.request
 
sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path("tinker://<unique_id>/sampler_weights/final")
checkpoint_archive_url_response = future.result()
 
# `checkpoint_archive_url_response.url` is a signed URL that can be downloaded
# until checkpoint_archive_url_response.expires
urllib.request.urlretrieve(checkpoint_archive_url_response.url, "archive.tar")