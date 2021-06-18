import copy
import os
import sys
import time
from datetime import datetime as datetime
from multiprocessing import Process
from threading import Thread

import cv2
import numpy as np


def dict2obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict2obj(j))
        elif isinstance(j, seqs):
            setattr(top, i, type(j)(dict2obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


def softmax(x: np.ndarray, axis: int):
    x1 = np.exp(x)
    return x1 / np.sum(x1, axis=axis)


class T:

    @staticmethod
    def concur(funcs: list, args: list, mode=1):
        """DOES NOT support lambdas as `funcs`!"""
        assert mode in [0, 1]
        Carrier = Thread if mode == 0 else Process
        threads = list()
        for func, arg in zip(funcs, args):
            p = Carrier(target=func, args=arg)
            p.start()
            threads.append(p)
        print('[parallel]', threads)
        for p in threads:
            p.join()

    @staticmethod
    def split(data: list, nworker: int):
        splits = list()
        num = len(data)
        size = num // nworker
        for i in range(nworker):
            s, e = i * size, min((i + 1) * size, num)
            splits.append(data[s:e])
        return splits


def nms_np(dets, thresh, mode="Union"):
    """TODO Reimplement and move into class B!
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :param mode:
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            raise NotImplementedError
        # keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def ensure_fold(path: str):
    if type(path) != str:
        raise NotImplementedError
    if not os.path.exists(path):
        os.makedirs(path)


def backup_fold(path: str, format='.%Y%m%d-%H%M%S'):
    assert os.path.isdir(path) and path.endswith('/')
    if len(os.listdir(path)) == 0:
        os.rmdir(path)
    else:
        os.rename(path, path[:-1] + time.strftime(format, time.localtime(os.path.getctime(path))))
    os.mkdir(path)


def print_progress(text, current, total, step=500):
    """If total is not known, then input 0."""
    if (current % step == 0) or (current == total) or (current + 1 == total):
        print('\r>> %s %d/%d' % (text, current, total))


def log_time_func():
    return '[%s %s()]' % (datetime.now().strftime('%Y%m%d%H%M'), sys._getframe().f_back.f_code.co_name)


def draw_result(img, bboxs, ldmks, fps=None, color=(255, 0, 255)):
    bboxs = [] if bboxs is None else bboxs
    for info in bboxs:
        bbox = list(map(int, info[:4]))
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color)
    ldmks = [] if ldmks is None else ldmks
    for ldmk in ldmks:
        ldmk = list(map(int, ldmk))
        for i in range(int(len(ldmk) / 2)):
            cv2.circle(img, (ldmk[2 * i], ldmk[2 * i + 1]), 1, color, thickness=-1)
    if fps is not None:
        fps = '{:.3f}'.format(fps)
        cv2.putText(img, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


"""I/B/L are utility classes for data processing. All of them DO NOT affect the input data!"""


class I:
    """Input image must be of HWC format and the number is one."""

    @staticmethod
    def crop_resize(im: np.ndarray, bbox: np.ndarray, size: int):
        """Only support one bounding bbox as input for now!"""
        assert (len(im.shape) == 3 or len(im.shape) == 2) and len(bbox.shape) == 1
        if bbox.dtype != np.int:
            bbox = bbox.astype(np.int)
        crop = im[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        return cv2.resize(crop, (size, size))

    @staticmethod
    def flip(im: np.ndarray):
        """Flip horizontally `im`."""
        assert len(im.shape) == 3 or len(im.shape) == 2
        return cv2.flip(im, 1)

    @staticmethod
    def rotate(im: np.ndarray, center: tuple, angle: int):
        """Rotate `im` around `center` by `angle` degrees"""
        assert (len(im.shape) == 3 or len(im.shape) == 2) and len(center) == 2
        mat = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        return cv2.warpAffine(copy.deepcopy(im), mat, (im.shape[1], im.shape[0]))

    @staticmethod
    def imgs2bulk(imgs: list):
        hws = np.array([img.shape[:2] for img in imgs])
        hwm = np.max(hws, axis=0)
        bulk = np.zeros((len(imgs), *hwm, 3,), imgs[0].dtype)
        for i, img in enumerate(imgs):
            bulk[i, :hws[i, 0], :hws[i, 1], :] = img
        return bulk, hws

    @staticmethod
    def img2tiers(img, size, scale0, factor, max_tier=99):
        tiers, scales = list(), list()
        wh = np.array(img.shape[:2][::-1]) * scale0
        tier0 = cv2.resize(img, tuple(wh.astype(np.int)))
        scale = scale0
        tiers.append(tier0)
        scales.append(scale)
        wh *= factor
        cnt = 1
        while min(wh) > size and cnt < max_tier:
            tier = cv2.resize(img, tuple(wh.astype(np.int)))
            scale *= factor
            tiers.append(tier)
            scales.append(scale)
            wh *= factor
            cnt += 1
        return tiers, scales


class B:
    """Bounding boxes' coordinates are [[left, top, right, bottom],..]!"""

    @staticmethod
    def rand_bbox(bgts: np.ndarray, num: int, cs=1.0, rs=(0.8, 1.25)):
        """Generate squares relative to `bgts`. Return tensor of shape (n*t, 4)."""
        assert len(bgts.shape) == 2  # (a, 4)  # t=bgts.shape[0], n=bgns.shape[0]=num
        assert bgts.shape[1] == 4
        cxy = np.mean(np.reshape(bgts, [-1, 2, 2]), 1)  # (t, 2)
        wh = bgts[:, 2:4] - bgts[:, 0:2]  # (t, 2)
        cx, cy = [B._rand_n(cxy[:, i], wh[:, i] * cs, num) for i in (0, 1)]  # (n, t)
        lu = np.sqrt(wh[:, 0] * wh[:, 1])[:, None] * np.array(rs)[None, :]  # (t, 2)
        r = B._rand_u(lu[:, 0], lu[:, 1], num) / 2  # (n, t)
        bgns = np.stack([cx - r, cy - r, cx + r, cy + r], -1).astype(np.int)  # (n, t, 4)
        return bgns.reshape([-1, 4])  # (n*t, 4)

    @staticmethod
    def _rand_u(lowers: np.ndarray, uppers: np.ndarray, nrow: int):
        """Generate uniform randoms of shape (`nrow`, `lowers.shape[0]`)."""
        assert len(lowers.shape) == 1 and len(uppers.shape) == 1
        return np.random.uniform(np.tile(lowers, [nrow, 1]), np.tile(uppers, [nrow, 1]))

    @staticmethod
    def _rand_n(means: np.ndarray, sigmas: np.ndarray, nrow: int):
        """Generate normal randoms of shape (`nrow`, `means.shape[0]`)."""
        assert len(means.shape) == 1 and len(sigmas.shape) == 1
        return np.random.normal(np.tile(means, [nrow, 1]), np.tile(sigmas, [nrow, 1]))

    @staticmethod
    def cate_bbox(bgns: np.ndarray, bgts: np.ndarray, thresh=(0.3, 0.4, 0.65)):  # TODO: check!!!!!!!!!!!!!!!!!!!!!!!!!!
        """Categorize `bgns` into pos/part/neg according to their IoU with `bgts`."""
        assert len(bgns.shape) == 2 and len(bgts.shape) == 2
        # n=bgns.shape[0], t=bgts.shape[0]
        ious_all = B.iou(bgns, bgts)  # (n, t)
        igts = np.arange(bgts.shape[0])[np.argmax(ious_all, axis=1)]  # (n,)
        ious = np.max(ious_all, axis=1)  # (n,)
        flag_pos = ious >= thresh[2]
        flag_part = np.logical_and(ious > thresh[1], ious < thresh[2])
        flag_neg = ious <= thresh[0]
        bdict_pos, bdict_part, bdict_neg = dict(), dict(), dict()
        for i in np.arange(bgts.shape[0]):
            flag_igt = igts == i
            bdict_pos.update({i: bgns[np.where(flag_pos & flag_igt)[0]]})
            bdict_part.update({i: bgns[np.where(flag_part & flag_igt)[0]]})
            bdict_neg.update({i: bgns[np.where(flag_neg & flag_igt)[0]]})
        return bdict_pos, bdict_part, bdict_neg

    @staticmethod
    def is_valid(bbox: np.ndarray, thresh=1):
        """Judge if `bbox`'s area >= `thresh`. Return tensor of shape (n, )."""
        assert len(bbox.shape) == 2
        wh = bbox[:, 2:4] - bbox[:, 0:2]
        return np.logical_and(wh[:, 0] >= thresh, wh[:, 1] >= thresh)

    @staticmethod
    def is_in(b1: np.ndarray, b2: np.ndarray):  # TODO: check
        """Judge if `b1` in `b2`."""
        assert len(b1.shape) == 2 and len(b2.shape) == 2
        assert b1.shape[1] == 4 and b2.shape[1] == 4
        c1 = np.logical_and(b1[:, 0::2] >= b2[:, 0], b1[:, 0::2] <= b2[:, 2])
        c2 = np.logical_and(b1[:, 1::2] >= b2[:, 1], b1[:, 1::2] <= b2[:, 3])
        return np.sum(c1, 1) + np.sum(c2, 1) == 4

    @staticmethod
    def ensure_in(b1: np.ndarray, b2: np.ndarray):  # TODO: check
        """Ensure `b1` in `b2`; not affect `b1`."""
        assert len(b1.shape) == 2 and len(b2.shape) == 2
        return np.hstack([np.maximum(b1[:, 0:2], b2[:, 0:2]), np.minimum(b1[:, 2:4], b2[:, 2:4])])

    @staticmethod
    def iou(b1: np.ndarray, b2: np.ndarray):
        """Calculate the Intersection-over-Union between `b1` and `b2`. Return tensor of shape (n1, n2).
        Support BATCH operation! ≧∇≦            XXX: implement GIoU."""
        assert len(b1.shape) == 2 and len(b2.shape) == 2  # (n1, 4), (n2, 4) -> (n1, n2)
        s1 = ((b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]))[:, None]  # (n1, 1)
        s2 = ((b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]))[None, :]  # (1, n2)
        b1, b2 = b1[:, None, :], b2[None, :, :]  # (n1, 1, 4), (1, n2, 4) -> inter: (n1, n2, 4)
        inter = np.concatenate([np.maximum(b1[..., 0:2], b2[..., 0:2]), np.minimum(b1[..., 2:4], b2[..., 2:4])], 2)
        zero = np.where(~B.is_valid(inter.reshape([-1, 4])).reshape(inter.shape[:2]))  # (n1, n2)
        si = (inter[:, :, 2] - inter[:, :, 0]) * (inter[:, :, 3] - inter[:, :, 1])  # (n1, n2)
        si[zero[0], zero[1]] = 0
        return si / (s1 + s2 - si)

    @staticmethod
    def transform(bbox: np.ndarray, base: np.ndarray):
        """Calculate the offsets of `bbox` relative to `base`:
            `bbox_gt` -- `bbox_gn` => `ofst`!
        """
        assert len(bbox.shape) == 2 and len(base.shape) == 2
        wh = base[:, 2:4] - base[:, 0:2]  # B.wh(base)
        return (bbox - base) / np.tile(wh, [1, 2])

    @staticmethod
    def transform_inverse(base: np.ndarray, ofst: np.ndarray):
        """Calculate the original bounding box according to `base` and `ofst`:
            `bbox_gn` ++ `ofst` => `bbox_gt`.
        """
        assert len(base.shape) == 2 and len(ofst.shape) == 2
        assert base.shape[1] == 4 and ofst.shape[1] == 4
        wh = base[:, 2:4] - base[:, 0:2]
        return (base + np.tile(wh, [1, 2]) * ofst).astype(base.dtype)

    @staticmethod
    def flip(bbox: np.ndarray, im_w: np.ndarray):  # TODO: check
        """Flip horizontally `bbox` relative to an image box [0, 0, `im_w`, `im_h`]."""
        assert len(bbox.shape) == 2 and len(im_w.shape) == 1
        assert bbox.shape[1] == 4
        b2 = np.copy(bbox)
        b2[:, 0::2] = im_w[:, None] - bbox[:, 0::2]
        return b2[:, [2, 1, 0, 3]]

    '''
    @staticmethod
    def expand2square(bbox: np.ndarray):
        """Expand `bbox` to square with the same center. Return tensor of shape (n, 4)."""
        assert len(bbox.shape) == 2
        cxcy = np.mean(np.reshape(bbox[:, 0:4], [-1, 2, 2]), 1)
        size_half = (np.max(bbox[:, 2:4] - bbox[:, 0:2], 1) / 2)[:, None]
        bbox2 = np.hstack([cxcy - size_half, cxcy + size_half])
        return bbox2.astype(bbox.dtype)
    '''

    @staticmethod
    def expand2square(bbox: np.ndarray, scale: float = 1.0):
        """Expand `bbox` to square with the same center. Return tensor of shape (n, 4)."""
        assert len(bbox.shape) == 2
        cxcy = np.mean(np.reshape(bbox[:, 0:4], [-1, 2, 2]), 1)
        size_half = (np.max(bbox[:, 2:4] - bbox[:, 0:2], 1) / 2)[:, None]
        if scale != 1.0:
            size_half *= scale
        bbox2 = np.hstack([cxcy - size_half, cxcy + size_half])
        return bbox2.astype(bbox.dtype)


class L:
    """Landmarks' coordinates are [[xyxyxyxyxy],..]!"""

    @staticmethod
    def is_in(ldmk: np.ndarray, bbox: np.ndarray):
        """Judge if `ldmk` in `bbox`; designed for filtering landmarks, which may not be needed.
        """
        assert len(ldmk.shape) == 2 and len(bbox.shape) == 2
        c1 = np.logical_and(ldmk[:, 0::2] >= bbox[:, 0], ldmk[:, 0::2] < bbox[:, 2])
        c2 = np.logical_and(ldmk[:, 1::2] >= bbox[:, 1], ldmk[:, 1::2] < bbox[:, 3])
        return np.sum(c1, 1) + np.sum(c2, 1) == ldmk.shape[1]

    @staticmethod
    def transform(ldmk: np.ndarray, bbox: np.ndarray):
        """Calculate the offsets of `ldmk` relative to `bbox`:
            `ldmk_gt` -- `bbox_gn` => `ofst`!
        """
        assert len(ldmk.shape) == 2 and len(bbox.shape) == 2
        wh = bbox[:, 2:4] - bbox[:, 0:2]  # B.wh(bbox)
        return (ldmk - np.tile(bbox[:, :2], [1, 5])) / np.tile(wh, [1, 5])

    @staticmethod
    def transform_inverse(bbox: np.ndarray, ofst: np.ndarray):
        """Calculate the original landmark according to `bbox` and `ofst`.
            `bbox_gn` ++ `ofst` => `ldmk_gt`.
        """
        assert len(bbox.shape) == 2 and len(ofst.shape) == 2
        wh = bbox[:, 2:4] - bbox[:, 0:2]  # B.wh(bbox)
        return (np.tile(bbox[:, :2], [1, 5]) + np.tile(wh, [1, 5]) * ofst).astype(np.int)

    @staticmethod
    def flip(ldmk: np.ndarray, im_w: np.ndarray):
        """Flip horizontally `ldmk` relative to an image box [0, 0, `im_w`, `im_h`].
        """
        assert len(ldmk.shape) == 2 and len(im_w.shape) == 1
        ldmk = np.copy(ldmk)
        ldmk[:, 0::2] = im_w[:, None] - ldmk[:, 0::2]
        return ldmk[:, [2, 3, 0, 1, 4, 5, 8, 9, 6, 7]]

    @staticmethod
    def rotate(ldmk: np.ndarray, center: tuple, angle: int):
        """Rotate `ldmk` around `center` by `angle` degrees.
        """
        assert len(ldmk.shape) == 2 and len(center) == 2
        ldmk = np.copy(ldmk)
        ldmk = ldmk.reshape([-1, 5, 2])
        mat = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        l2 = np.sum(ldmk[:, :, None, :] * mat[:, :2][None, None, ...], 3) + mat[:, 2][None, None, :]
        return l2.reshape([-1, 10]).astype(np.int)
