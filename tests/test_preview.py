from dmotion.detector import Detection, mirror_detections


def test_preview_mirroring_preserves_original_detections():
    original = Detection((10, 20, 110, 220), 0.8, "banknotes")
    mirrored = mirror_detections([original], width=640)
    assert mirrored == [Detection((530, 20, 630, 220), 0.8, "banknotes")]
    assert original.box == (10, 20, 110, 220)
    assert mirror_detections(mirrored, width=640) == [original]
