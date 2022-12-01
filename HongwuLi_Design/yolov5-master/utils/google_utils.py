# Google utils: https://cloud.google.com/storage/docs/reference/libraries


import os, platform, subprocess, time
from pathlib import Path
import requests, torch

def gsutil_getsize(url=''):
    s = subprocess.check_output(f"gsutil du {url}", shell=True).decode('utf-8')
    if len(s):
        return eval(s.split(' ')[0])
    else:
        return 0


def attempt_download(file, repo='ultralytics/yolov5'):
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']
        except:
            assets = [
             'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
             'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=(subprocess.STDOUT)).decode().split()[(-1)]
            except:
                tag = 'v5.0'

        name = file.name
        if name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/"
            redundant = False
            try:
                try:
                    url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
                    print(f"Downloading {url} to {file}...")
                    torch.hub.download_url_to_file(url, file)
                    assert file.exists() and file.stat().st_size > 1000000.0
                except Exception as e:
                    print(f"Download error: {e}")
                    assert redundant, 'No secondary mirror'
                    url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                    print(f"Downloading {url} to {file}...")
                    os.system(f"curl -L '{url}' -o '{file}' --retry 3 -C -")

            finally:
                return

            if not file.exists() or file.stat().st_size < 1000000.0:
                file.unlink(missing_ok=True)
                print(f"ERROR: Download failure: {msg}")
            print('')
            return


def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    t = time.time()
    file = Path(file)
    cookie = Path('cookie')
    print(f"Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ", end='')
    file.unlink(missing_ok=True)
    cookie.unlink(missing_ok=True)
    out = 'NUL' if platform.system() == 'Windows' else '/dev/null'
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)
    cookie.unlink(missing_ok=True)
    if r != 0:
        file.unlink(missing_ok=True)
        print('Download error ')
        return r
    else:
        if file.suffix == '.zip':
            print('unzipping... ', end='')
            os.system(f"unzip -q {file}")
            file.unlink()
        print(f"Done ({time.time() - t:.1f}s)")
        return r


def get_token(cookie='./cookie'):
    with open(cookie) as (f):
        for line in f:
            if 'download' in line:
                return line.split()[(-1)]

    return ''