import argparse
import os
import os.path as osp
import shutil
import time
import warnings
import numpy as np
import tempfile
from pathlib import Path

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmcv.image import tensor2imgs

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
import models
from mmseg.datasets.custom import *

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


def _dataset_filename(dataset, idx):
    for attr in ('img_infos', 'data_infos'):
        if hasattr(dataset, attr):
            infos = getattr(dataset, attr)
            if isinstance(infos, list) and idx < len(infos) and isinstance(infos[idx], dict):
                for key in ('filename', 'file_name', 'img_path', 'img'):
                    if key in infos[idx]:
                        return osp.basename(str(infos[idx][key]))
    if hasattr(dataset, 'get_img_info'):
        info = dataset.get_img_info(idx)
        if isinstance(info, dict):
            for key in ('filename', 'file_name'):
                if key in info:
                    return osp.basename(str(info[key]))
    return f'{idx:06d}.png'


def _find_vis_file(show_dir, filename):
    # mmseg usually saves as: out_dir / ori_filename (often .jpg)
    candidates = [
        osp.join(show_dir, filename),
        osp.join(show_dir, osp.basename(filename)),
    ]
    stem = Path(filename).stem
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
        candidates.append(osp.join(show_dir, stem + ext))

    for p in candidates:
        if osp.exists(p):
            return p

    # Fallback: recursive search by basename/stem.
    show_root = Path(show_dir)
    base = Path(filename).name
    for p in show_root.rglob(base):
        if p.is_file():
            return str(p)
    for p in show_root.rglob(stem + '.*'):
        if p.is_file():
            return str(p)
    return None


def _extract_topk_labels(seg, ignore_index, topk=8, min_pixels=300):
    labels, counts = np.unique(seg, return_counts=True)
    keep = []
    for label, count in zip(labels.tolist(), counts.tolist()):
        if ignore_index is not None and int(label) == int(ignore_index):
            continue
        if count < int(min_pixels):
            continue
        keep.append((int(label), int(count)))
    keep.sort(key=lambda x: x[1], reverse=True)
    return [label for label, _ in keep[: int(topk)]]


def _label_anchor(seg, label):
    ys, xs = np.where(seg == label)
    if ys.size == 0:
        return None

    try:
        import cv2  # type: ignore

        mask = (seg == label).astype(np.uint8)
        num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            x1, y1 = int(xs.min()), int(ys.min())
            return x1, y1

        # stats: [label, x, y, w, h, area], label 0 is background
        areas = stats[1:, 4]
        j = int(np.argmax(areas)) + 1
        x, y, w, h = stats[j, 0], stats[j, 1], stats[j, 2], stats[j, 3]
        return int(x), int(y)
    except Exception:
        return int(xs.min()), int(ys.min())


def _draw_class_tags(img_path, seg, class_names, seen_idx, unseen_idx, out_path,
                     ignore_index=None, topk=8, min_pixels=300, font_size=16):
    if Image is None:
        raise RuntimeError('PIL is required for --tag-classes (pip install pillow).')

    labels = _extract_topk_labels(seg, ignore_index, topk=topk, min_pixels=min_pixels)
    if not labels:
        if out_path != img_path:
            mmcv.copyfile(img_path, out_path)
        return

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype('arial.ttf', int(font_size))
    except Exception:
        font = ImageFont.load_default()

    def text_wh(text):
        # Pillow compatibility: ImageDraw.textbbox may require TrueType fonts.
        try:
            if hasattr(draw, 'textbbox'):
                x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
                return (x1 - x0), (y1 - y0)
        except Exception:
            pass
        try:
            if hasattr(font, 'getbbox'):
                x0, y0, x1, y1 = font.getbbox(text)
                return (x1 - x0), (y1 - y0)
        except Exception:
            pass
        try:
            if hasattr(draw, 'textsize'):
                return draw.textsize(text, font=font)
        except Exception:
            pass
        try:
            return font.getsize(text)
        except Exception:
            return (len(text) * int(font_size * 0.6), int(font_size))

    seen_set = set(int(x) for x in (seen_idx or []))
    unseen_set = set(int(x) for x in (unseen_idx or []))

    occupied = []

    def place_rect(x0, y0, w, h):
        x0 = max(0, min(int(x0), img.width - w - 1))
        y0 = max(0, min(int(y0), img.height - h - 1))
        for _ in range(30):
            rect = (x0, y0, x0 + w, y0 + h)
            if all(rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3] for r in occupied):
                occupied.append(rect)
                return rect
            y0 = min(img.height - h - 1, y0 + h + 2)
        rect = (x0, y0, x0 + w, y0 + h)
        occupied.append(rect)
        return rect

    for label in labels:
        name = str(class_names[label]) if class_names and label < len(class_names) else str(label)
        anchor = _label_anchor(seg, label) or (5, 5)

        text_color = (0, 0, 0)
        if label in unseen_set:
            text_color = (220, 0, 0)  # red for unseen
        elif label in seen_set:
            text_color = (0, 0, 0)  # black for seen

        pad = 3
        tw, th = text_wh(name)
        w, h = tw + 2 * pad, th + 2 * pad

        rect = place_rect(anchor[0], anchor[1], w, h)
        draw.rectangle(rect, fill=(255, 255, 255))
        draw.text((rect[0] + pad, rect[1] + pad), name, font=font, fill=text_color)

    mmcv.mkdir_or_exist(osp.dirname(out_path))
    img.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--tag-classes',
        action='store_true',
        help='draw in-image class tags (paper-like qualitative figure). Requires --show-dir.')
    parser.add_argument('--tag-topk', type=int, default=8, help='max number of tags per image')
    parser.add_argument('--tag-min-pixels', type=int, default=300, help='min pixels for a class to be tagged')
    parser.add_argument('--tag-font-size', type=int, default=16, help='font size for tags')
    parser.add_argument(
        '--tag-out-dir',
        default=None,
        help='optional output dir for tagged images (default: overwrite --show-dir results)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    # setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0: #do this way
        # work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])
        work_dir = os.path.join(('/').join(args.checkpoint.split('/')[:-1]),'eval_results')
        print('work_dir:', work_dir)
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if 'CLIP' in cfg.model.type:
        cfg.model.class_names = list(dataset.CLASSES)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if hasattr(model, 'presegmentor'):
        model.presegmentor.init_weights() # to initial conv1/2 to get pseudo mask
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if cfg.model['test_cfg']['zsmode'] == 'both':
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    # elif cfg.model['test_cfg']['zsmode'] == 'novel':
    #     model.CLASSES = ()
    #     model.PALETTE = ()
    #     for i in cfg.model['novel_class']:
    #         model.CLASSES = model.CLASSES + (checkpoint['meta']['CLASSES'][i],)
    #         model.PALETTE = model.PALETTE + (checkpoint['meta']['PALETTE'][i],)

    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE

    seen_idx = cfg.model.base_class
    unseen_idx = cfg.model.novel_class
    
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    #     model.CLASSES = dataset.CLASSES
    # if 'PALETTE' in checkpoint.get('meta', {}):
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    # else:
    #     print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    #     model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
        if args.tag_classes:
            if not args.show_dir:
                raise ValueError('--tag-classes requires --show-dir')
            tag_out_dir = args.tag_out_dir or args.show_dir
            mmcv.mkdir_or_exist(tag_out_dir)
            ignore_index = getattr(dataset, 'ignore_index', None)
            for i, seg in enumerate(results):
                filename = _dataset_filename(dataset, i)
                vis_path = _find_vis_file(args.show_dir, filename)
                if not vis_path:
                    continue
                out_path = osp.join(tag_out_dir, osp.basename(vis_path))
                _draw_class_tags(
                    vis_path,
                    seg,
                    getattr(dataset, 'CLASSES', None),
                    seen_idx,
                    unseen_idx,
                    out_path,
                    ignore_index=ignore_index,
                    topk=args.tag_topk,
                    min_pixels=args.tag_min_pixels,
                    font_size=args.tag_font_size,
                )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    print('test done')
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(seen_idx, unseen_idx, results, **eval_kwargs) ##modify this part (Custom Evaluation)
            # metric = MyCustomDataset.evaluate(results, **eval_kwargs) ##modify this part (Custom Evaluation)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
