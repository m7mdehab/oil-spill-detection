"""Tests for the canonical metrics module.

The core cases use tiny tensors whose confusion matrix and resulting IoU /
precision / recall / F1 are computed by hand in the comments, so the test
asserts against independently-derived numbers rather than the code's own output.
"""

import math

import torch

from oilspill.metrics import (
    NUM_CLASSES,
    OIL_CLASS_INDEX,
    SegmentationMetrics,
    compute_metrics,
    logits_to_labels,
)

REL = 1e-6


def test_perfect_prediction() -> None:
    """pred == target ⇒ every present-class metric is 1.0, absent classes nan."""
    target = torch.tensor([[0, 0], [1, 1]])
    result = compute_metrics(target.clone(), target)

    assert result.iou[0] == 1.0
    assert result.iou[1] == 1.0
    # classes 2,3,4 never appear -> undefined
    for c in (2, 3, 4):
        assert math.isnan(result.iou[c])
        assert math.isnan(result.precision[c])
    assert result.mean_iou == 1.0
    assert result.macro_f1 == 1.0
    assert result.pixel_accuracy == 1.0


def test_hand_computed_confusion() -> None:
    """A 2x2 example with a single misclassification, fully hand-derived.

    target = [[0, 0],     pred = [[0, 1],
              [1, 1]]              [1, 1]]

    Confusion matrix (rows=truth, cols=pred), only classes 0,1 populated:
        C[0,0]=1, C[0,1]=1, C[1,1]=2
    Class 0: TP=1, FP=0, FN=1 -> IoU=1/2,  P=1,    R=1/2,  F1=2/3
    Class 1: TP=2, FP=1, FN=0 -> IoU=2/3,  P=2/3,  R=1,    F1=4/5
    mIoU       = mean(1/2, 2/3)        = 7/12
    macro P    = mean(1,   2/3)        = 5/6
    macro R    = mean(1/2, 1)          = 3/4
    macro F1   = mean(2/3, 4/5)        = 11/15
    pixel acc  = 3 correct / 4 total   = 3/4
    """
    target = torch.tensor([[0, 0], [1, 1]])
    preds = torch.tensor([[0, 1], [1, 1]])
    r = compute_metrics(preds, target)

    assert r.confusion_matrix[0, 0] == 1
    assert r.confusion_matrix[0, 1] == 1
    assert r.confusion_matrix[1, 1] == 2

    assert math.isclose(r.iou[0].item(), 1 / 2, rel_tol=REL)
    assert math.isclose(r.iou[1].item(), 2 / 3, rel_tol=REL)
    assert math.isclose(r.precision[0].item(), 1.0, rel_tol=REL)
    assert math.isclose(r.precision[1].item(), 2 / 3, rel_tol=REL)
    assert math.isclose(r.recall[0].item(), 1 / 2, rel_tol=REL)
    assert math.isclose(r.recall[1].item(), 1.0, rel_tol=REL)
    assert math.isclose(r.f1[0].item(), 2 / 3, rel_tol=REL)
    assert math.isclose(r.f1[1].item(), 4 / 5, rel_tol=REL)

    assert math.isclose(r.mean_iou, 7 / 12, rel_tol=REL)
    assert math.isclose(r.macro_precision, 5 / 6, rel_tol=REL)
    assert math.isclose(r.macro_recall, 3 / 4, rel_tol=REL)
    assert math.isclose(r.macro_f1, 11 / 15, rel_tol=REL)
    assert math.isclose(r.pixel_accuracy, 3 / 4, rel_tol=REL)


def test_ignore_index_excludes_pixels() -> None:
    """Ignored pixels must not enter the confusion matrix at all.

    target = [[0, 255],   pred = [[0, 1],
              [1, 1  ]]            [1, 1]]
    With ignore_index=255 the top-right pixel is dropped, leaving a perfect
    match on the remaining three pixels:
        C[0,0]=1, C[1,1]=2  -> every present-class metric = 1.0, mIoU = 1.0
    Without ignoring it, class 0 would gain an FN and class 1 an FP, so this
    test also guards against the ignore handling silently doing nothing.
    """
    target = torch.tensor([[0, 255], [1, 1]])
    preds = torch.tensor([[0, 1], [1, 1]])

    r = compute_metrics(preds, target, ignore_index=255)
    assert r.confusion_matrix.sum().item() == 3
    assert math.isclose(r.iou[0].item(), 1.0, rel_tol=REL)
    assert math.isclose(r.iou[1].item(), 1.0, rel_tol=REL)
    assert math.isclose(r.mean_iou, 1.0, rel_tol=REL)

    # Sanity: not ignoring changes the answer (proves the ignore did something).
    r_noignore = compute_metrics(
        torch.tensor([[0, 1], [1, 1]]),
        torch.tensor([[0, 2], [1, 1]]),  # use class 2 instead of 255
    )
    assert not math.isclose(r_noignore.mean_iou, 1.0, rel_tol=REL)


def test_all_wrong_off_by_one() -> None:
    """Predict the wrong class everywhere ⇒ zero IoU on every involved class.

    target all 0, pred all 1:
        C[0,1] = N. Class 0: TP=0,FP=0,FN=N -> IoU=0,recall=0,precision nan.
        Class 1: TP=0,FP=N,FN=0 -> IoU=0,precision=0,recall nan.
    """
    target = torch.zeros(3, 3, dtype=torch.long)
    preds = torch.ones(3, 3, dtype=torch.long)
    r = compute_metrics(preds, target)

    assert r.iou[0].item() == 0.0
    assert r.iou[1].item() == 0.0
    assert r.recall[0].item() == 0.0
    assert math.isnan(r.precision[0].item())  # class 0 never predicted
    assert r.precision[1].item() == 0.0
    assert math.isnan(r.recall[1].item())  # class 1 never in truth
    assert r.mean_iou == 0.0
    assert r.pixel_accuracy == 0.0


def test_streaming_matches_one_shot() -> None:
    """Accumulating two batches equals computing on their concatenation."""
    torch.manual_seed(0)
    p1 = torch.randint(0, NUM_CLASSES, (4, 4))
    t1 = torch.randint(0, NUM_CLASSES, (4, 4))
    p2 = torch.randint(0, NUM_CLASSES, (4, 4))
    t2 = torch.randint(0, NUM_CLASSES, (4, 4))

    metric = SegmentationMetrics()
    metric.update(p1, t1)
    metric.update(p2, t2)
    streamed = metric.compute()

    one_shot = compute_metrics(torch.cat([p1, p2]), torch.cat([t1, t2]))
    assert torch.equal(streamed.confusion_matrix, one_shot.confusion_matrix)
    assert math.isclose(streamed.mean_iou, one_shot.mean_iou, rel_tol=REL)


def test_reset_clears_state() -> None:
    metric = SegmentationMetrics()
    metric.update(torch.ones(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))
    metric.reset()
    metric.update(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))
    assert metric.compute().iou[0].item() == 1.0


def test_logits_to_labels() -> None:
    # (N=1, C=3, H=1, W=2): pick class 2 then class 0.
    logits = torch.tensor([[[[0.1, 5.0]], [[0.2, 0.3]], [[9.0, 0.1]]]])
    labels = logits_to_labels(logits)
    assert labels.shape == (1, 1, 2)
    assert labels[0, 0, 0].item() == 2
    assert labels[0, 0, 1].item() == 0


def test_shape_mismatch_raises() -> None:
    metric = SegmentationMetrics()
    try:
        metric.update(torch.zeros(2, 2, dtype=torch.long), torch.zeros(3, 3, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_to_dict_is_json_friendly() -> None:
    import json

    target = torch.tensor([[0, 0], [1, 1]])
    preds = torch.tensor([[0, 1], [1, 1]])
    d = compute_metrics(preds, target).to_dict()
    # round-trips through json and absent classes are null, not 0/1
    text = json.dumps(d)
    back = json.loads(text)
    assert back["per_class"]["iou"]["Look-alike"] is None
    assert back["aggregate"]["oil_iou"] == d["aggregate"]["oil_iou"]  # type: ignore[index]
    assert OIL_CLASS_INDEX == 1
