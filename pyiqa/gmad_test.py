"""Run a candidate-pool gMAD-style comparison between two NR-IQA metrics."""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import torch

from pyiqa.api_helpers import create_metric
from pyiqa.default_model_configs import DEFAULT_CONFIGS


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def _collect_images(image_dir):
    """Return supported image files below a directory in stable path order."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError(f'Image directory does not exist: {image_dir}')

    images = sorted(
        path.resolve()
        for path in image_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) < 2:
        raise ValueError(f'At least two supported images are required in {image_dir}')
    return images


def _score_images(metric_name, images, device):
    """Run one configured NR metric once per image and return scalar scores."""
    metric_config = DEFAULT_CONFIGS.get(metric_name)
    if metric_config is None:
        raise ValueError(f'Unknown pyiqa metric: {metric_name}')
    if metric_config.get('metric_mode') != 'NR':
        raise ValueError(f'gMAD candidate-pool scoring requires an NR metric: {metric_name}')

    metric = create_metric(metric_name, device=device).eval()
    scores = []
    with torch.inference_mode():
        for image_path in images:
            score = metric(str(image_path)).reshape(-1)
            if score.numel() != 1:
                raise ValueError(
                    f'Metric {metric_name} returned {score.numel()} scores for {image_path}'
                )
            value = float(score.item())
            if not math.isfinite(value):
                raise ValueError(f'Metric {metric_name} returned a non-finite score for {image_path}')
            scores.append(value)

    return metric, scores


def _load_score_csv(scores_csv, metric_a, metric_b, image_root=None):
    """Load two score columns and resolve each row's image path.

    The CSV must contain ``image`` and columns named after both selected
    metrics. Relative paths are interpreted from ``image_root`` when supplied,
    otherwise from the directory containing the CSV.
    """
    scores_csv = Path(scores_csv).resolve()
    root = Path(image_root).resolve() if image_root else scores_csv.parent
    images = []
    scores_a = []
    scores_b = []

    with scores_csv.open(newline='', encoding='utf-8-sig') as score_file:
        reader = csv.DictReader(score_file)
        required = {'image', metric_a, metric_b}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                f'{scores_csv} must have CSV columns image, {metric_a}, {metric_b}; '
                f'missing: {", ".join(missing)}'
            )

        seen_images = set()
        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get('image') or '').strip()
            if not image_value:
                raise ValueError(f'Missing image path in {scores_csv} at row {row_number}')
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = root / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise ValueError(f'Image from {scores_csv} row {row_number} does not exist: {image_path}')
            if image_path in seen_images:
                raise ValueError(f'Duplicate image path in {scores_csv} at row {row_number}: {image_path}')
            seen_images.add(image_path)

            try:
                value_a = float(row[metric_a])
                value_b = float(row[metric_b])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f'Invalid score in {scores_csv} at row {row_number}; '
                    f'{metric_a} and {metric_b} must be numeric'
                ) from error
            if not math.isfinite(value_a) or not math.isfinite(value_b):
                raise ValueError(f'Non-finite score in {scores_csv} at row {row_number}')

            images.append(image_path)
            scores_a.append(value_a)
            scores_b.append(value_b)

    if len(images) < 2:
        raise ValueError(f'At least two scored images are required in {scores_csv}')
    return images, scores_a, scores_b


def _resolve_lower_better(metric_name, override):
    """Resolve score direction from an explicit override or pyiqa metadata."""
    if override is not None:
        return override
    metric_config = DEFAULT_CONFIGS.get(metric_name)
    if metric_config is None:
        raise ValueError(
            f'Unknown metric direction for {metric_name}; provide --direction-a/--direction-b '
            'as lower or higher'
        )
    return metric_config.get('lower_better', False)


def _standardize(scores, lower_better):
    """Orient higher quality as better, then z-score scores within the pool."""
    oriented = [-score if lower_better else score for score in scores]
    mean = sum(oriented) / len(oriented)
    variance = sum((score - mean)**2 for score in oriented) / len(oriented)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return [0.0] * len(oriented)
    return [(score - mean) / std for score in oriented]


def _best_pair(tie_scores, target_scores, tolerance):
    """Find the greatest target-score gap among pairs tied by one metric.

    A sliding window over sorted tie scores limits candidate pairs to the
    requested tolerance; monotonic queues track target-score extrema per window.
    """
    ordered = sorted(range(len(tie_scores)), key=tie_scores.__getitem__)
    minimum = deque()
    maximum = deque()
    right = 1
    best = None

    for left in range(len(ordered) - 1):
        while minimum and minimum[0][0] <= left:
            minimum.popleft()
        while maximum and maximum[0][0] <= left:
            maximum.popleft()

        right = max(right, left + 1)
        while right < len(ordered) and (
            tie_scores[ordered[right]] - tie_scores[ordered[left]] <= tolerance
        ):
            value = target_scores[ordered[right]]
            while minimum and minimum[-1][1] >= value:
                minimum.pop()
            minimum.append((right, value))
            while maximum and maximum[-1][1] <= value:
                maximum.pop()
            maximum.append((right, value))
            right += 1

        if not minimum:
            continue

        first_index = ordered[left]
        for extreme_position, extreme_score in (minimum[0], maximum[0]):
            gap = abs(target_scores[first_index] - extreme_score)
            if best is None or gap > best['target_gap']:
                second_index = ordered[extreme_position]
                tie_gap = abs(tie_scores[first_index] - tie_scores[second_index])
                best = {
                    'first_index': first_index,
                    'second_index': second_index,
                    'tie_gap': tie_gap,
                    'target_gap': gap,
                }
    return best


def _pair_result(pair, images, scores_a, scores_b, normalized_a, normalized_b, name_a, name_b):
    """Format a selected pair with raw and standardized metric scores."""
    if pair is None:
        return None

    first = pair['first_index']
    second = pair['second_index']
    return {
        'images': [
            {
                'path': str(images[index]),
                name_a: scores_a[index],
                name_b: scores_b[index],
                f'{name_a}_standardized': normalized_a[index],
                f'{name_b}_standardized': normalized_b[index],
            }
            for index in (first, second)
        ],
        'standardized_tie_gap': pair['tie_gap'],
        'standardized_target_gap': pair['target_gap'],
    }


def run_gmad(
    image_dir=None,
    metric_a='musiq',
    metric_b='brisque',
    tie_tolerance=0.1,
    device=None,
    scores_csv=None,
    image_root=None,
    lower_better_a=None,
    lower_better_b=None,
):
    """Find candidate pairs where either metric ties while the other differs.

    Supply either ``image_dir`` for pyiqa inference or ``scores_csv`` containing
    precomputed scores. The function standardizes within this candidate pool;
    its output is a gMAD-style exploratory result, not a complete official
    gMAD evaluation.
    """
    if tie_tolerance < 0:
        raise ValueError('tie_tolerance must be non-negative')
    if metric_a == metric_b:
        raise ValueError('metric_a and metric_b must identify different metrics/score columns')

    if scores_csv:
        if image_dir:
            raise ValueError('Provide either image_dir or scores_csv, not both')
        images, scores_a, scores_b = _load_score_csv(
            scores_csv, metric_a, metric_b, image_root=image_root
        )
        lower_better_a = _resolve_lower_better(metric_a, lower_better_a)
        lower_better_b = _resolve_lower_better(metric_b, lower_better_b)
        source = 'precomputed scores'
    else:
        if not image_dir:
            raise ValueError('Provide image_dir or scores_csv')
        if image_root:
            raise ValueError('image_root can only be used with scores_csv')
        images = _collect_images(image_dir)
        model_a, scores_a = _score_images(metric_a, images, device)
        model_b, scores_b = _score_images(metric_b, images, device)
        lower_better_a = model_a.lower_better
        lower_better_b = model_b.lower_better
        source = 'pyiqa inference'

    normalized_a = _standardize(scores_a, lower_better_a)
    normalized_b = _standardize(scores_b, lower_better_b)

    pair_a_ties = _best_pair(normalized_a, normalized_b, tie_tolerance)
    pair_b_ties = _best_pair(normalized_b, normalized_a, tie_tolerance)

    return {
        'method': 'candidate-pool gMAD-style search',
        'score_source': source,
        'image_count': len(images),
        'tie_tolerance_standard_deviations': tie_tolerance,
        'metrics': {
            metric_a: {'lower_better': lower_better_a},
            metric_b: {'lower_better': lower_better_b},
        },
        f'{metric_a}_ties_{metric_b}_differs': _pair_result(
            pair_a_ties,
            images,
            scores_a,
            scores_b,
            normalized_a,
            normalized_b,
            metric_a,
            metric_b,
        ),
        f'{metric_b}_ties_{metric_a}_differs': _pair_result(
            pair_b_ties,
            images,
            scores_a,
            scores_b,
            normalized_a,
            normalized_b,
            metric_a,
            metric_b,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Find image pairs from a candidate directory where one NR-IQA metric '
            'agrees and another differs. This is a gMAD-style helper, not the '
            'official gMAD evaluation protocol.'
        )
    )
    parser.add_argument(
        'image_dir',
        nargs='?',
        help='Directory of candidate images (searched recursively)',
    )
    parser.add_argument(
        '--scores-csv',
        help='CSV with image, <metric-a>, and <metric-b> columns containing precomputed scores',
    )
    parser.add_argument('--metric-a', default='musiq', help='First NR-IQA metric (default: musiq)')
    parser.add_argument('--metric-b', default='brisque', help='Second NR-IQA metric (default: brisque)')
    parser.add_argument(
        '--image-root',
        help='Root used to resolve relative image paths in --scores-csv (default: CSV directory)',
    )
    parser.add_argument(
        '--direction-a',
        choices=('lower', 'higher'),
        help='Score direction for metric A in CSV mode (inferred for known pyiqa metrics)',
    )
    parser.add_argument(
        '--direction-b',
        choices=('lower', 'higher'),
        help='Score direction for metric B in CSV mode (inferred for known pyiqa metrics)',
    )
    parser.add_argument(
        '--tie-tolerance',
        type=float,
        default=0.1,
        help='Maximum standardized score gap for the tied metric (default: 0.1)',
    )
    parser.add_argument('--device', default=None, help='Inference device, e.g. cuda or cpu')
    parser.add_argument('-o', '--output', help='Write JSON results to this file (default: stdout)')
    args = parser.parse_args()

    results = run_gmad(
        args.image_dir,
        args.metric_a,
        args.metric_b,
        tie_tolerance=args.tie_tolerance,
        device=args.device,
        scores_csv=args.scores_csv,
        image_root=args.image_root,
        lower_better_a=None if args.direction_a is None else args.direction_a == 'lower',
        lower_better_b=None if args.direction_b is None else args.direction_b == 'lower',
    )
    output = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output + '\n', encoding='utf-8')
    else:
        print(output)


if __name__ == '__main__':
    main()
