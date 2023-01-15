import os
import re
import requests
from bs4 import BeautifulSoup


data_dir = "/Users/nealdigre/xprize/datasets/toucans/Yellow-throated Toucan (Chestnut-mandibled)/"
url_calls = "https://search.macaulaylibrary.org/catalog?taxonCode=chmtou1&sort=rating_rank_desc&tag=call&view=list"
url_songs = "https://search.macaulaylibrary.org/catalog?taxonCode=chmtou1&sort=rating_rank_desc&tag=song&view=list"


def download_macaulay_audio(html, subset='calls'):
    soup = BeautifulSoup(html).find_all('li')
    for lineitem in soup:
        match = re.search(r"ML[0-9]*", lineitem.text)
        if match:
            fileid = match.group().removeprefix('ML')
            data = requests.get(f"https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{fileid}/audio")
            if data:
                with open(os.path.join(data_dir, subset, f'{fileid}.mp3'), 'wb') as f:
                    f.write(data.content)


html_calls = requests.get(url_calls).content
html_songs = requests.get(url_songs).content

download_macaulay_audio(html_calls, 'calls')
download_macaulay_audio(html_songs, 'songs')
