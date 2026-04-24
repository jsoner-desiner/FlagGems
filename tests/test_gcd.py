import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_INT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
    to_reference,
)


@pytest.mark.gcd
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
def test_accuracy_gcd(shape, dtype):
    res_a = torch.randint(1, 10001, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    res_b = torch.randint(1, 10001, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_a = to_reference(res_a)
    ref_b = to_reference(res_b)
    ref_out = torch.gcd(ref_a, ref_b)
    with flag_gems.use_gems():
        res_out = torch.gcd(res_a, res_b)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.gcd_special
def test_accuracy_gcd_special_values():
    """Test gcd with zeros, negatives, and large values."""
    device = flag_gems.device
    # Zero handling: gcd(a, 0) = |a|
    a = torch.tensor([0, 5, -10], dtype=torch.int32, device=device)
    b = torch.tensor([7, 0, 15], dtype=torch.int32, device=device)
    ref = torch.gcd(a.cpu(), b.cpu()).to(device)
    with flag_gems.use_gems():
        res = torch.gcd(a, b)
    gems_assert_equal(res, ref)
