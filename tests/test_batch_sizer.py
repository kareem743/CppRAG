import pytest

from rag.batch_sizer import AdaptiveBatchSizer


def test_batch_sizer_adjusts_with_backpressure() -> None:
    sizer = AdaptiveBatchSizer(current=100, min_size=10, max_size=200, target_seconds=10.0)

    sizer.record(25.0, had_error=False)
    assert sizer.current < 100

    previous = sizer.current
    sizer.record(2.0, had_error=False)
    assert sizer.current >= previous
    after_fast = sizer.current

    sizer.record(5.0, had_error=True)
    assert sizer.current <= after_fast


def test_batch_sizer_clamps_current_on_init() -> None:
    sizer_low = AdaptiveBatchSizer(current=1, min_size=10, max_size=200, target_seconds=5.0)
    assert sizer_low.current == 10

    sizer_high = AdaptiveBatchSizer(current=500, min_size=10, max_size=200, target_seconds=5.0)
    assert sizer_high.current == 200


def test_batch_sizer_validates_inputs() -> None:
    with pytest.raises(ValueError):
        AdaptiveBatchSizer(current=10, min_size=0, max_size=10, target_seconds=5.0)
    with pytest.raises(ValueError):
        AdaptiveBatchSizer(current=10, min_size=5, max_size=4, target_seconds=5.0)
    with pytest.raises(ValueError):
        AdaptiveBatchSizer(current=10, min_size=5, max_size=20, target_seconds=0.0)


def test_batch_sizer_record_respects_bounds() -> None:
    sizer = AdaptiveBatchSizer(current=10, min_size=10, max_size=12, target_seconds=10.0)
    sizer.record(1.0, had_error=False)
    assert sizer.current == 12
    sizer.record(30.0, had_error=False)
    assert sizer.current == 10
