import click
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
import clip


@click.command()
@click.option('--image_in', type=click.File('rb'))
@click.option('--embedding_out', type=click.File('wb'))
@click.option('--processes', type=int)
def clipper(image_in, embedding_out, processes):
    # load model in the master process first to avoid downloading
    # the model from multiple processes in parallel
    _, preprocess = clip.load('ViT-B/32', device='cpu')

    # Image.fromarray assumes the input is laid-out as unsigned 8-bit integers.
    images = np.load(image_in)
    images = torch.stack(list(map(lambda x: preprocess(Image.fromarray(x).convert('RGB')), images)))
    images = torch.split(images, (images.shape[0] + processes - 1) // processes)

    with mp.Pool(processes) as pool:
        embeddings = pool.map(extract_feature, images)
    pool.join()

    embedding = np.concatenate(embeddings)
    np.save(embedding_out, embedding)


def extract_feature(image):
    model, _ = clip.load('ViT-B/32', device='cpu') # force inference on CPU

    # CLIP can only process one image at a time
    with torch.inference_mode():
        embedding = model.encode_image(image)
        embedding = embedding.detach().cpu().numpy()

    return embedding


if __name__ == '__main__':
    mp.set_start_method('forkserver')
    clipper()
