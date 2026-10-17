import pytest
import torch

from pyiqa.archs.func_util import extract_2d_patches
from pyiqa.archs.pieapp_arch import PieAPP


@pytest.fixture(scope='module')
def shared_model():
    torch.manual_seed(7)
    return PieAPP(pretrained=False).eval()


@pytest.fixture
def model(shared_model):
    shared_model.patch_batch_size = 64
    yield shared_model
    shared_model.patch_batch_size = 64


@pytest.mark.parametrize('patch_batch_size', [0, -1, 1.5, True, '64'])
def test_patch_batch_size_must_be_positive_integer_or_none(patch_batch_size):
    with pytest.raises(
        ValueError, match='patch_batch_size must be a positive integer or None'
    ):
        PieAPP(pretrained=False, patch_batch_size=patch_batch_size)


def test_default_patch_batch_size_is_64(model):
    assert model.patch_batch_size == 64


def test_patch_grid_is_a_view(model):
    image = torch.rand(2, 3, 91, 118)

    patch_grid = model._extract_patch_grid(image)

    assert patch_grid.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    ('batch_size', 'height', 'width', 'patch_batch_size'),
    [
        (1, 64, 64, 8),
        (1, 91, 118, 4),
        (2, 91, 118, 5),
    ],
)
def test_streamed_patches_match_full_unfold(
    model, batch_size, height, width, patch_batch_size
):
    image = torch.rand(batch_size, 3, height, width)
    expected = extract_2d_patches(
        image, model.patch_size, model.stride, padding='none'
    ).flatten(0, 1)
    patch_grid = model._extract_patch_grid(image)

    actual = torch.cat(
        [
            model._get_patch_batch(
                patch_grid,
                start,
                min(start + patch_batch_size, len(expected)),
            )
            for start in range(0, len(expected), patch_batch_size)
        ]
    )

    assert torch.equal(actual, expected)


def test_streamed_patch_batching_preserves_scores_across_image_batches(model):
    distorted = torch.rand(2, 3, 91, 118)
    reference = torch.rand(2, 3, 91, 118)

    with torch.inference_mode():
        model.patch_batch_size = None
        expected = model(distorted, reference)
        model.patch_batch_size = 5
        actual = model(distorted, reference)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_patch_batching_does_not_change_training_mode(model, monkeypatch):
    distorted = torch.rand(3, 3, 64, 64)
    reference = torch.rand(3, 3, 64, 64)
    model.train()

    try:
        with torch.inference_mode():
            model.patch_batch_size = None
            expected = model(distorted, reference)

            processed_batch_sizes = []
            compute_patch_outputs = model._compute_patch_outputs

            def record_batch_size(dist_patches, ref_patches):
                processed_batch_sizes.append(len(dist_patches))
                return compute_patch_outputs(dist_patches, ref_patches)

            monkeypatch.setattr(
                model, '_compute_patch_outputs', record_batch_size
            )
            model.patch_batch_size = 2
            actual = model(distorted, reference)
    finally:
        model.eval()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert processed_batch_sizes == [3]


def test_streamed_patch_batching_preserves_input_gradients(model):
    parameter_gradient_settings = [
        parameter.requires_grad for parameter in model.parameters()
    ]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    distorted = torch.rand(1, 3, 64, 91, requires_grad=True)
    reference = torch.rand(1, 3, 64, 91, requires_grad=True)

    try:
        model.patch_batch_size = None
        expected_score = model(distorted, reference)
        expected_gradients = torch.autograd.grad(
            expected_score.sum(), (distorted, reference)
        )

        model.patch_batch_size = 1
        actual_score = model(distorted, reference)
        actual_gradients = torch.autograd.grad(
            actual_score.sum(), (distorted, reference)
        )
    finally:
        for parameter, requires_grad in zip(
            model.parameters(), parameter_gradient_settings
        ):
            parameter.requires_grad_(requires_grad)

    torch.testing.assert_close(actual_score, expected_score, rtol=1e-5, atol=1e-6)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6
        )
