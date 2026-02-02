import tinker
import urllib.request
from dotenv import load_dotenv

WEIGHTS_PATH="tinker://81a400fc-92f1-5497-a59d-12bbc695e174:train:0/sampler_weights/final"
OUTPUT_PATH="weights/step_final.tar"

load_dotenv()
sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path(WEIGHTS_PATH)
checkpoint_archive_url_response = future.result()
 
# `checkpoint_archive_url_response.url` is a signed URL that can be downloaded
# until checkpoint_archive_url_response.expires
urllib.request.urlretrieve(checkpoint_archive_url_response.url, OUTPUT_PATH) 