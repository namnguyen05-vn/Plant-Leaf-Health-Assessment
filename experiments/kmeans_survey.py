"""H4.1 independent survey. Run: python -m experiments.kmeans_survey --synthetic.

RGB is uint8. HSV uses OpenCV H=0..179 rescaled with round(H*255/179),
S,V=0..255. Euclidean hue distance ignores its circular seam. Lab uses
OpenCV uint8 encoding: L*=L8*100/255, a*=a8-128, b*=b8-128; encoded
distances are not physical Lab distances. Never rank color spaces by SSE.
All area percentages use ROI pixels; full-image mode includes background.
Timings exclude visualization/IO and are not full pipeline timings.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import warnings

import cv2
import numpy as np

from src.segmentation.adapter import segment_leaf


def configurations(init='random'):
    if init not in ('random', 'k-means++'):
        raise ValueError('init must be random or k-means++')
    return [dict(color_space=space, k=k, init=init)
            for space in ('RGB', 'HSV', 'Lab') for k in range(2, 6)]


def validate_image(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
            or rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape):
        raise ValueError('Expected nonempty uint8 RGB (H,W,3)')


def convert_features(rgb, space):
    validate_image(rgb)
    if space == 'RGB':
        return rgb.copy()
    if space == 'HSV':
        features = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        features[..., 0] = np.rint(features[..., 0].astype(float) * 255 / 179).astype(np.uint8)
        return features
    if space == 'Lab':
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    raise ValueError('Unknown color space')


def validate_roi(rgb, roi):
    validate_image(rgb)
    if roi is None:
        return np.ones(rgb.shape[:2], bool)
    if (not isinstance(roi, np.ndarray) or roi.shape != rgb.shape[:2]
            or roi.dtype.kind not in 'buif' or not np.isfinite(roi).all()):
        raise ValueError('ROI must be finite numeric/bool HxW')
    valid = roi > 0
    if valid.sum() < 5:
        raise ValueError('ROI must contain at least five pixels for K=2..5')
    return valid


def inertia(features, labels, centers, roi):
    valid = (roi > 0) & (labels >= 0)
    residual = features[valid].astype(np.float64) - centers[labels[valid]].astype(np.float64)
    return float(np.sum(residual * residual))


def mask_metrics(result, roi):
    valid = roi > 0
    coverage = np.zeros(valid.shape, np.uint8)
    values = {}
    for name in ('disease', 'healthy', 'uncertain'):
        mask = result[name + '_mask']
        if (mask.shape != valid.shape or mask.dtype != np.uint8
                or not np.isin(mask, [0, 255]).all()):
            raise ValueError('Invalid mask contract')
        selected = mask > 0
        if np.any(selected & ~valid):
            raise ValueError('Mask outside ROI')
        coverage += selected
        count = int(selected.sum())
        values[name + '_pixel_count'] = count
        values[name + '_ratio_percent'] = 100 * count / int(valid.sum())
    if not np.array_equal(coverage, ((result['labels_2d'] >= 0) & valid).astype(np.uint8)):
        raise ValueError('Masks overlap or do not partition valid labels')
    if not np.array_equal(result['labels_2d'] >= 0, valid):
        raise ValueError('Labels do not preserve ROI')
    return values


def synthetic_scene():
    y, x = np.indices((96, 128))
    roi = ((x - 64) / 53) ** 2 + ((y - 48) / 39) ** 2 <= 1
    rgb = np.full((96, 128, 3), 235, np.uint8)
    rgb[roi] = [40, 145, 65]
    rgb[roi & (x > 76) & (y < 46)] = [205, 190, 40]
    rgb[roi & (x < 50) & (y > 48)] = [125, 70, 25]
    return rgb, roi


def read_image(path, mask=False):
    try:
        image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8),
                             cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    except OSError as exc:
        raise ValueError(f'Cannot read image: {path}') from exc
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image if mask else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(path, image):
    # Exclusive file creation also protects against accidental overwrites.
    encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                           if image.ndim == 3 else image)[1]
    with Path(path).open('xb') as stream:
        stream.write(encoded.tobytes())


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_survey(rgb, output_dir, roi=None, image_id='image', init='random',
               random_state=42, max_iter=100, tol=1e-3, repeats=1):
    configs = configurations(init)
    valid = validate_roi(rgb, roi)
    if valid.sum() < 5:
        raise ValueError('At least five pixels required')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if roi is None:
        warnings.warn('Full-image mode: background is clustered; image is not asserted to be leaf.', stacklevel=2)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_image(output / 'original.png', rgb)
    if roi is not None:
        write_image(output / 'roi.png', valid.astype(np.uint8) * 255)
    rows, summaries, tiles = [], [], []
    for config in configs:
        group = []
        for repeat in range(1, repeats + 1):
            start = perf_counter()
            features = convert_features(rgb, config['color_space'])
            conversion_ms = (perf_counter() - start) * 1000
            result = segment_leaf(rgb, valid, features, k=config['k'], init=init,
                                  random_state=random_state, max_iter=max_iter, tol=tol)
            row = dict(image_id=image_id, **config, random_state=random_state,
                       max_iter=max_iter, tol=tol, repeat=repeat,
                       roi_mode='full_image' if roi is None else 'provided',
                       image_height=rgb.shape[0], image_width=rgb.shape[1],
                       roi_pixel_count=int(valid.sum()), n_iter=result['n_iter'],
                       conversion_exec_time_ms=conversion_ms,
                       kmeans_exec_time_ms=result['exec_time_ms'],
                       lesion_exec_time_ms=result['lesion_exec_time_ms'],
                       adapter_exec_time_ms=result['adapter_exec_time_ms'],
                       inertia=inertia(features, result['labels_2d'], result['centroids'], valid),
                       **mask_metrics(result, valid))
            if any(not np.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
                raise ValueError('Nonfinite survey metric')
            rows.append(row)
            group.append(row)
        summary = dict(group[0], repeat='median', benchmark_repeats=repeats)
        for key in group[0]:
            if key.endswith('_ms'):
                summary[key] = median(item[key] for item in group)
        summaries.append(summary)
        folder = output / f"{config['color_space']}_k{config['k']}_{init}"
        folder.mkdir()
        segmented = result['kmeans_result']['segmented_img']
        write_image(folder / 'segmented_rgb.png', segmented)
        for name in ('disease', 'healthy', 'uncertain'):
            write_image(folder / f'{name}_mask.png', result[name + '_mask'])
        overlay = rgb.copy()
        selection = result['disease_mask'] > 0
        overlay[selection] = np.rint(.5 * rgb[selection] + .5 * np.array([255, 0, 0])).astype(np.uint8)
        write_image(folder / 'disease_overlay.png', overlay)
        tile = np.full((rgb.shape[0] + 28, rgb.shape[1], 3), 255, np.uint8)
        tile[28:] = segmented
        cv2.putText(tile, f"{config['color_space']} K={config['k']}", (3, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
        tiles.append(tile)
    write_image(output / f'comparison_{init}.png', np.concatenate([
        np.concatenate(tiles[i:i+4], axis=1) for i in range(0, 12, 4)], axis=0))
    write_csv(output / 'results.csv', rows)
    write_csv(output / 'summary.csv', summaries)
    with (output / 'notes.txt').open('x', encoding='utf-8') as stream:
        stream.write(__doc__ + '\nSummary timings are medians; other metrics use the same fixed seed.\n'
                     + json.dumps(dict(init=init, random_state=random_state, max_iter=max_iter,
                                       tol=tol, repeats=repeats, roi_mode=rows[0]['roi_mode'])))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--input', type=Path, help='Image file or directory (nonrecursive)')
    parser.add_argument('--roi', type=Path, help='Mask file, or directory containing <image_stem>.png')
    parser.add_argument('--output', type=Path, required=True, help='New directory; must not exist')
    parser.add_argument('--init', choices=['random', 'k-means++'], default='random')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    kwargs = dict(init=args.init, random_state=args.random_state, max_iter=args.max_iter,
                  tol=args.tol, repeats=args.repeats)
    if args.synthetic:
        if args.roi:
            parser.error('--roi is not used with --synthetic')
        rgb, roi = synthetic_scene()
        run_survey(rgb, args.output, roi, image_id='synthetic', **kwargs)
    elif args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in
                       ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        if not files:
            parser.error('No supported images in input directory')
        if args.roi and not args.roi.is_dir():
            parser.error('Directory input requires a directory of ROI masks')
        args.output.mkdir(parents=True, exist_ok=False)
        for index, path in enumerate(files):
            roi = read_image(args.roi / (path.stem + '.png'), True) if args.roi else None
            run_survey(read_image(path), args.output / f'{index:04d}_{path.stem}',
                       roi, image_id=path.name, **kwargs)
    else:
        run_survey(read_image(args.input), args.output,
                   read_image(args.roi, True) if args.roi else None,
                   image_id=args.input.name, **kwargs)


if __name__ == '__main__':
    main()
