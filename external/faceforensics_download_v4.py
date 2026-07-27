#!/usr/bin/env python
""" Downloads FaceForensics++ and Deep Fake Detection public data release
Example usage:
    see -h or https://github.com/ondyari/FaceForensics
"""
# -*- coding: utf-8 -*-
import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join
import shutil


# URLs and filenames
FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5',]

# Parameters
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
    }
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics v2 public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        help='Which dataset to download, either pristine or '
                             'manipulated data or the downloaded youtube '
                             'videos.',
                        choices=list(DATASETS.keys()) + ['all']
                        )
    parser.add_argument('-c', '--compression', type=str, default='raw',
                        help='Which compression degree. All videos '
                             'have been generated with h264 with a varying '
                             'codec. Raw (c0) videos are lossless compressed.',
                        choices=COMPRESSION
                        )
    parser.add_argument('-t', '--type', type=str, default='videos',
                        help='Which file type, i.e. videos, masks, for our '
                             'manipulation methods, models, for Deepfakes.',
                        choices=TYPE
                        )
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Select a number of videos number to '
                             "download if you don't want to download the full"
                             ' dataset.')
    parser.add_argument('--server', type=str, default='EU2',
                        help='Server to download the data from. Default set to EU2 mirror (kaldir.vc.in.tum.de).',
                        choices=SERVERS
                        )
    args = parser.parse_args()

    # URLs
    server = args.server
    if server == 'EU2':
        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'
    elif server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'
    elif server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'
    else:
        raise Exception('Wrong server name. Choices: {}'.format(str(SERVERS)))
    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = server_url + 'v3/manipulated_sequences/' + \
                               'Deepfakes/models/'

    return args


def fetch_url(url, retries=5, delay=2):
    # Try given URL first, then try alternate server mirrors if connection is reset
    urls_to_try = [url]
    if 'canis.vc.in.tum.de:8100' in url:
        urls_to_try.append(url.replace('http://canis.vc.in.tum.de:8100/', 'http://kaldir.vc.in.tum.de/faceforensics/'))
        urls_to_try.append(url.replace('http://canis.vc.in.tum.de:8100/', 'http://falas.cmpt.sfu.ca:8100/'))
    elif 'kaldir.vc.in.tum.de/faceforensics' in url:
        urls_to_try.append(url.replace('http://kaldir.vc.in.tum.de/faceforensics/', 'http://falas.cmpt.sfu.ca:8100/'))
        urls_to_try.append(url.replace('http://kaldir.vc.in.tum.de/faceforensics/', 'http://canis.vc.in.tum.de:8100/'))

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    last_err = None
    for u in urls_to_try:
        req = urllib.request.Request(u, headers=headers)
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.read().decode("utf-8")
            except Exception as e:
                last_err = e
                time.sleep(delay)
    raise last_err


def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    if report_progress:
        filenames = tqdm(filenames)
    for filename in filenames:
        download_file(base_url + filename, join(output_path, filename))


def download_file(url, out_file, report_progress=False, retries=3):
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(out_file):
        tqdm.write('WARNING: skipping download of existing file ' + out_file)
        return

    urls_to_try = [url]
    if 'canis.vc.in.tum.de:8100' in url:
        urls_to_try.append(url.replace('http://canis.vc.in.tum.de:8100/', 'http://kaldir.vc.in.tum.de/faceforensics/'))
        urls_to_try.append(url.replace('http://canis.vc.in.tum.de:8100/', 'http://falas.cmpt.sfu.ca:8100/'))
    elif 'kaldir.vc.in.tum.de/faceforensics' in url:
        urls_to_try.append(url.replace('http://kaldir.vc.in.tum.de/faceforensics/', 'http://canis.vc.in.tum.de:8100/'))
        urls_to_try.append(url.replace('http://kaldir.vc.in.tum.de/faceforensics/', 'http://falas.cmpt.sfu.ca:8100/'))
    elif 'falas.cmpt.sfu.ca:8100' in url:
        urls_to_try.append(url.replace('http://falas.cmpt.sfu.ca:8100/', 'http://kaldir.vc.in.tum.de/faceforensics/'))

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    last_err = None
    for target_url in urls_to_try:
        for attempt in range(retries):
            out_file_tmp = None
            try:
                fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
                os.close(fh)
                req = urllib.request.Request(target_url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(out_file_tmp, 'wb') as out_f:
                    shutil.copyfileobj(resp, out_f)
                os.rename(out_file_tmp, out_file)
                return
            except Exception as e:
                last_err = e
                if out_file_tmp and os.path.exists(out_file_tmp):
                    try:
                        os.remove(out_file_tmp)
                    except Exception:
                        pass
                time.sleep(1)
    tqdm.write(f"ERROR: Failed to download {url}: {last_err}")
    raise last_err


def main(args):
    # TOS
    if not getattr(args, "yes", False):
        print('By pressing any key to continue you confirm that you have agreed '\
              'to the FaceForensics terms of use as described at:')
        print(args.tos_url)
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        _ = input('')

    # Extract arguments
    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    c_type = args.type
    c_compression = args.compression
    num_videos = args.num_videos
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # Check for special dataset cases
    for dataset in c_datasets:
        dataset_path = DATASETS[dataset]
        # Special cases
        if 'original_youtube_videos' in dataset:
            # Here we download the original youtube videos zip file
            print('Downloading original youtube videos.')
            if not 'info' in dataset_path:
                print('Please be patient, this may take a while (~40gb)')
                suffix = ''
            else:
                suffix = 'info'
            download_file(args.base_url + '/' + dataset_path,
                          out_file=join(output_path,
                                        'downloaded_videos{}.zip'.format(
                                            suffix)),
                          report_progress=True)
            return

        # Else: regular datasets
        print('Downloading {} of dataset "{}"'.format(
            c_type, dataset_path
        ))

        # Get filelists and video lenghts list from server
        if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
            filepaths = json.loads(fetch_url(args.base_url + '/' + DEEPFEAKES_DETECTION_URL))
            if 'actors' in dataset_path:
                filelist = filepaths['actors']
            else:
                filelist = filepaths['DeepFakesDetection']
        elif 'original' in dataset_path:
            # Load filelist from server
            file_pairs = json.loads(fetch_url(args.base_url + '/' + FILELIST_URL))
            filelist = []
            for pair in file_pairs:
                filelist += pair
        else:
            # Load filelist from server
            file_pairs = json.loads(fetch_url(args.base_url + '/' + FILELIST_URL))
            # Get filelist
            filelist = []
            for pair in file_pairs:
                filelist.append('_'.join(pair))
                if c_type != 'models':
                    filelist.append('_'.join(pair[::-1]))
        # Maybe limit number of videos for download
        if num_videos is not None and num_videos > 0:
        	print('Downloading the first {} videos'.format(num_videos))
        	filelist = filelist[:num_videos]

        # Server and local paths
        dataset_videos_url = args.base_url + '{}/{}/{}/'.format(
            dataset_path, c_compression, c_type)
        dataset_mask_url = args.base_url + '{}/{}/videos/'.format(
            dataset_path, 'masks', c_type)

        if c_type == 'videos':
            dataset_output_path = join(output_path, dataset_path, c_compression,
                                       c_type)
            print('Output path: {}'.format(dataset_output_path))
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_videos_url, dataset_output_path)
        elif c_type == 'masks':
            dataset_output_path = join(output_path, dataset_path, c_type,
                                       'videos')
            print('Output path: {}'.format(dataset_output_path))
            if 'original' in dataset:
                if args.dataset != 'all':
                    print('Only videos available for original data. Aborting.')
                    return
                else:
                    print('Only videos available for original data. '
                          'Skipping original.\n')
                    continue
            if 'FaceShifter' in dataset:
                print('Masks not available for FaceShifter. Aborting.')
                return
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_mask_url, dataset_output_path)

        # Else: models for deepfakes
        else:
            if dataset != 'Deepfakes' and c_type == 'models':
                print('Models only available for Deepfakes. Aborting')
                return
            dataset_output_path = join(output_path, dataset_path, c_type)
            print('Output path: {}'.format(dataset_output_path))

            # Get Deepfakes models
            for folder in tqdm(filelist):
                folder_filelist = DEEPFAKES_MODEL_NAMES

                # Folder paths
                folder_base_url = args.deepfakes_model_url + folder + '/'
                folder_dataset_output_path = join(dataset_output_path,
                                                  folder)
                download_files(folder_filelist, folder_base_url,
                               folder_dataset_output_path,
                               report_progress=False)   # already done


if __name__ == "__main__":
    args = parse_args()
    main(args)
