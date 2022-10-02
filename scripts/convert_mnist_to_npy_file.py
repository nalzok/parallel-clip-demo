import gzip
import urllib.request
import numpy as np
import pathlib


if __name__ == '__main__':
    url = 'http://yann.lecun.com/exdb/mnist'
    path = 'data'
    for kind in ['train', 't10k']:
        labels_name = f'{kind}-labels-idx1-ubyte.gz'
        images_name = f'{kind}-images-idx3-ubyte.gz'

        labels_url = f'{url}/{labels_name}'
        labels_path = pathlib.Path(f'{path}/{labels_name}')
        images_url = f'{url}/{images_name}'
        images_path = pathlib.Path(f'{path}/{images_name}')

        labels_npy = pathlib.Path(f'{path}/{kind}_labels.npy')
        images_npy = pathlib.Path(f'{path}/{kind}_images.npy')

        if labels_npy.exists() and images_npy.exists():
            continue

        # Downloading
        if not labels_path.exists():
            print(f'Downloading {labels_url}...')
            urllib.request.urlretrieve(labels_url, labels_path)

        if not images_path.exists():
            print(f'Downloading {images_url}...')
            urllib.request.urlretrieve(images_url, images_path)


        # Decompressing
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 28, 28)


        # Saving
        print(f'Saving {labels_npy}...')
        np.save(labels_npy, labels)

        print(f'Saving {images_npy}...')
        np.save(images_npy, images)
