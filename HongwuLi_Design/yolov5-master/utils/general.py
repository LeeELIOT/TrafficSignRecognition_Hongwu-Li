# YOLOv5 general utils


import glob, logging, math, os, platform, random, re, subprocess, time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import cv2, numpy as np, pandas as pd, pkg_resources as pkg, torch, torchvision, yaml
from utils.google_utils import gsutil_getsize
from utils.metrics import fitness
from utils.torch_utils import init_torch_seeds
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
pd.options.display.max_columns = 10
cv2.setNumThreads(0)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))

def set_logging(rank=-1, verbose=True):
    logging.basicConfig(format='%(message)s',
      level=logging.INFO if (verbose and rank in (-1, 0)) else (logging.WARN))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    if last_list:
        return max(last_list, key=(os.path.getctime))
    else:
        return ''


def is_docker():
    return Path('/workspace').exists()


def is_colab():
    try:
        import google.colab
        return True
    except Exception as e:
        return False


def is_pip():
    return 'site-packages' in Path(__file__).absolute().parts


def emojis(str=''):
    if platform.system() == 'Windows':
        return str.encode().decode('ascii', 'ignore')
    else:
        return str


def file_size(file):
    return Path(file).stat().st_size / 1000000.0


def check_online():
    import socket
    try:
        socket.create_connection(('1.1.1.1', 443), 5)
        return True
    except OSError:
        return False


def check_git_status():
    print((colorstr('github: ')), end='')
    try:
        if not Path('.git').exists():
            raise AssertionError('skipping check (not a git repository)')
        else:
            if not not is_docker():
                raise AssertionError('skipping check (Docker image)')
            elif not check_online():
                raise AssertionError('skipping check (offline)')
            cmd = 'git fetch && git config --get remote.origin.url'
            url = subprocess.check_output(cmd, shell=True).decode().strip().rstrip('.git')
            branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()
            n = int(subprocess.check_output(f"git rev-list {branch}..origin/master --count", shell=True))
            if n > 0:
                s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. Use 'git pull' to update or 'git clone {url}' to download latest."
            else:
                s = f"up to date with {url} ✅"
        print(emojis(s))
    except Exception as e:
        print(e)


def check_python(minimum='3.6', required=True):
    current = platform.python_version()
    result = pkg.parse_version(current) >= pkg.parse_version(minimum)
    if required:
        if not result:
            raise AssertionError(f"Python {minimum} required by YOLOv5, but Python {current} is currently installed")
    return result


def check_requirements(requirements='requirements.txt', exclude=()):
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        requirements = [f"{x.name}{x.specifier}" for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:
        requirements = [x for x in requirements if x not in exclude]
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:
            n += 1
            print(f"{prefix} {r} not found and is required by YOLOv5, attempting auto-update...")
            try:
                print(subprocess.check_output(f"pip install '{r}'", shell=True).decode())
            except Exception as e:
                print(f"{prefix} {e}")

    if n:
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))


def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_imshow():
    try:
        if not not is_docker():
            raise AssertionError('cv2.imshow() is disabled in Docker environments')
        elif not not is_colab():
            raise AssertionError('cv2.imshow() is disabled in Google Colab environments')
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f"WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}")
        return False


def check_file(file):
    file = str(file)
    if Path(file).is_file() or file == '':
        return file
    else:
        if file.startswith(('http://', 'https://')):
            url, file = file, Path(file).name
            print(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"
            return file
        else:
            files = glob.glob(('./**/' + file), recursive=True)
            assert len(files), f"File not found: {file}"
            assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]


def check_dataset(dict):
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in val if isinstance(val, list) else [val]]
        all(x.exists() for x in val) or print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
        if s and len(s):
            if s.startswith('http'):
                if s.endswith('.zip'):
                    f = Path(s).name
                    print(f"Downloading {s} ...")
                    torch.hub.download_url_to_file(s, f)
                    r = os.system(f"unzip -q {f} -d ../ && rm {f}")
            else:
                if s.startswith('bash '):
                    print(f"Running {s} ...")
                    r = os.system(s)
                else:
                    r = exec(s)
            print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))
        else:
            raise Exception('Dataset not found.')


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):

    def download_one(url, dir):
        f = dir / Path(url).name
        if not f.exists():
            print(f"Downloading {url} to {f}...")
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
            else:
                torch.hub.download_url_to_file(url, f, progress=True)
        else:
            if unzip:
                if f.suffix in ('.zip', '.gz'):
                    print(f"Unzipping {f}...")
                    if f.suffix == '.zip':
                        s = f"unzip -qo {f} -d {dir} && rm {f}"
                    else:
                        if f.suffix == '.gz':
                            s = f"tar xfz {f} --directory {f.parent}"
                    if delete:
                        s += f" && rm {f}"
                    os.system(s)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in tuple(url) if isinstance(url, str) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    return re.sub(pattern='[|@#!¡·$€%&()=?¿^*;:,¨´><+]', repl='_', string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: (1 - math.cos(x * math.pi / steps)) / 2 * (y2 - y1) + y1


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black':'\x1b[30m',  'red':'\x1b[31m', 
     'green':'\x1b[32m', 
     'yellow':'\x1b[33m', 
     'blue':'\x1b[34m', 
     'magenta':'\x1b[35m', 
     'cyan':'\x1b[36m', 
     'white':'\x1b[37m', 
     'bright_black':'\x1b[90m', 
     'bright_red':'\x1b[91m', 
     'bright_green':'\x1b[92m', 
     'bright_yellow':'\x1b[93m', 
     'bright_blue':'\x1b[94m', 
     'bright_magenta':'\x1b[95m', 
     'bright_cyan':'\x1b[96m', 
     'bright_white':'\x1b[97m', 
     'end':'\x1b[0m', 
     'bold':'\x1b[1m', 
     'underline':'\x1b[4m'}
    return ''.join(colors[x] for x in args) + (f"{string}") + colors['end']


def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return torch.Tensor()
    else:
        labels = np.concatenate(labels, 0)
        classes = labels[:, 0].astype(np.int)
        weights = np.bincount(classes, minlength=nc)
        weights[weights == 0] = 1
        weights = 1 / weights
        weights /= weights.sum()
        return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    class_counts = np.array([np.bincount((x[:, 0].astype(np.int)), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights


def coco80_to_coco91_class():
    x = [
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
     35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
     64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw
    y[:, 1] = h * x[:, 1] + padh
    return y


def segment2box(segment, width=640, height=640):
    x, y = segment.T
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y = x[inside], y[inside]
    if any(x):
        return np.array([x.min(), y.min(), x.max(), y.max()])
    else:
        return np.zeros((1, 4))


def segments2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])

    return xyxy2xywh(np.array(boxes))


def resample_segments(segments, n=1000):
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T

    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-07):
    box2 = box2.T
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = (
         box1[0], box1[1], box1[2], box1[3])
        b2_x1, b2_y1, b2_x2, b2_y2 = (box2[0], box2[1], box2[2], box2[3])
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            if CIoU:
                v = 4 / math.pi ** 2 * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    else:
        return iou


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def wh_iou(wh1, wh2):
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.min(wh1, wh2).prod(2)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5
    xc = prediction[(Ellipsis, 4)] > conf_thres
    if not 0 <= conf_thres <= 1:
        raise AssertionError(f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0")
    elif not 0 <= iou_thres <= 1:
        raise AssertionError(f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0")
    min_wh, max_wh = (2, 4096)
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    output = [torch.zeros((0, 6), device=(prediction.device))] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=(x.device))
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[(range(len(l)), l[:, 0].long() + 5)] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            pass
        else:
            x[:, 5:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[(i, j + 5, None)], j[:, None].float()), 1)
            else:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[(conf.view(-1) > conf_thres)]
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=(x.device))).any(1)]
            n = x.shape[0]
            if not n:
                continue
            else:
                if n > max_nms:
                    x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge:
            if 1 < n < 3000.0:
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[(iou.sum(1) > 1)]
            else:
                output[xi] = x[i]
                if time.time() - t > time_limit:
                    print(f"WARNING: NMS time limit {time_limit}s exceeded")
                    break

    return output


def strip_optimizer(f='best.pt', s=''):
    x = torch.load(f, map_location=(torch.device('cpu')))
    if x.get('ema'):
        x['model'] = x['ema']
    for k in ('optimizer', 'training_results', 'wandb_id', 'ema', 'updates'):
        x[k] = None

    x['epoch'] = -1
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False

    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1000000.0
    print(f"Optimizer stripped from {f},{' saved as %s,' % s if s else ''} {mb:.1f}MB")


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    a = '%10s' * len(hyp) % tuple(hyp.keys())
    b = '%10.3g' * len(hyp) % tuple(hyp.values())
    c = '%10.4g' * len(results) % results
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)
    with open('evolve.txt', 'a') as (f):
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)
    x = x[np.argsort(-fitness(x))]
    np.savetxt('evolve.txt', x, '%10.3g')
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[(0, i + 7)])

    with open(yaml_file, 'w') as (f):
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)
    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))


def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()
            b = xyxy2xywh(d[:, :4])
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
            b[:, 2:] = b[:, 2:] * 1.3 + 30
            d[:, :4] = xywh2xyxy(b).long()
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))
                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im, dtype=(np.float32))
                im /= 255.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)
            x[i] = x[i][(pred_cls1 == pred_cls2)]

    return x


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[(0, 1)]):int(xyxy[(0, 3)]), int(xyxy[(0, 0)]):int(xyxy[(0, 2)]), ::1 if BGR else -1]
    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)
    return crop


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists():
        if not exist_ok:
            suffix = path.suffix
            path = path.with_suffix('')
            dirs = glob.glob(f"{path}{sep}*")
            matches = [re.search(f"%s{sep}(\\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            path = Path(f"{path}{sep}{n}{suffix}")
    dir = path if path.suffix == '' else path.parent
    if not dir.exists():
        if mkdir:
            dir.mkdir(parents=True, exist_ok=True)
    return path