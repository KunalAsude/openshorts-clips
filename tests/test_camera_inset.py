import importlib

import camera_inset
import gpu_render
from camera_inset import (
    INSET_ASPECT,
    STREAMER_CAMERA_RATIO,
    gameplay_crop,
    inset_box,
    inset_filtergraph,
    streamer_filtergraph,
    is_cornered,
    nearest_corner,
    usable,
)

HD = (1920, 1080)


class TestNearestCorner:
    def test_reads_the_quadrant(self):
        assert nearest_corner([10, 10, 100, 100], *HD) == ("left", "top")
        assert nearest_corner([1800, 980, 100, 100], *HD) == ("right", "bottom")
        assert nearest_corner([1800, 10, 100, 100], *HD) == ("right", "top")
        assert nearest_corner([10, 980, 100, 100], *HD) == ("left", "bottom")


class TestIsCornered:
    def test_small_subject_against_an_edge_qualifies(self):
        # A webcam pinned to the bottom-right: the exact case YOLO reports as a
        # short, wide upper-body box.
        assert is_cornered([1500, 900, 350, 120], *HD)

    def test_a_presenter_filling_the_shot_is_rejected(self):
        assert not is_cornered([600, 100, 700, 900], *HD)

    def test_a_small_but_centred_subject_is_rejected(self):
        assert not is_cornered([900, 500, 150, 100], *HD)

    def test_box_not_touching_the_bottom_still_counts(self):
        # Measured failure: an inset pinned to the right edge sat 18% clear of
        # the bottom because the detector stopped at the chest. Requiring two
        # edges threw it away.
        assert is_cornered([991 * 1.5, 502 * 1.5, 246 * 1.5, 86 * 1.5], *HD)

    def test_a_tall_subject_is_never_an_inset(self):
        assert not is_cornered([1700, 100, 200, 800], *HD)


class TestInsetBox:
    def test_box_stays_inside_the_frame(self):
        for box in ([0, 980, 200, 100], [1800, 0, 120, 120],
                    [1850, 1000, 70, 80]):
            x, y, w, h = inset_box(box, *HD)
            assert 0 <= x and x + w <= 1920
            assert 0 <= y and y + h <= 1080

    def test_bottom_anchored_inset_reaches_the_bottom_edge(self):
        # The head sits above the detection, so the box must extend upward
        # while staying pinned to the edge the inset actually touches.
        x, y, w, h = inset_box([1500, 950, 300, 100], *HD)
        assert y + h == 1080
        assert y < 950

    def test_height_follows_the_16_9_assumption(self):
        # A wide, flat torso detection must still yield a webcam-shaped box.
        _x, _y, w, h = inset_box([1000, 800, 300, 80], *HD)
        assert h >= w / INSET_ASPECT - 1

    def test_box_grows_around_the_subject(self):
        x, y, w, h = inset_box([1600, 900, 200, 120], *HD)
        assert w >= 200 and h >= 120


class TestFaceDetections:
    """A face box grown like a torso gave a box that started below the
    forehead (Excel walkthrough, 17-oct-2026)."""

    def test_a_face_in_the_corner_keeps_the_whole_head(self):
        face = [1054, 497, 128, 128]          # measured, 1280x720
        x, y, w, h = inset_box(face, 1280, 720, face=True)
        assert y <= 497 - 0.3 * 128
        assert y + h == 720
        assert w >= 3 * 128 - 1

    def test_a_face_box_gives_a_webcam_shaped_box(self):
        _x, _y, w, h = inset_box([1088, 458, 96, 96], 1280, 720, face=True)
        assert abs(w / h - INSET_ASPECT) < 0.05

    def test_face_and_body_detections_agree(self):
        # Same webcam, one sample found the face and the next only the body:
        # the boxes must land on the same inset for stable_box to agree.
        by_face = inset_box([77, 845, 125, 125], 1920, 1080, face=True)
        by_body = inset_box([0, 838, 231, 96], 1920, 1080)
        cx = lambda b: b[0] + b[2] / 2.0
        assert abs(cx(by_face) - cx(by_body)) <= 1920 * 0.05


class TestUsable:
    def test_rejects_a_speck(self):
        assert not usable((0, 0, 40, 20), 1080)

    def test_rejects_something_that_fills_the_frame(self):
        assert not usable((0, 0, 1920, 900), 1080)

    def test_accepts_a_normal_webcam(self):
        assert usable((1500, 850, 350, 200), 1080)


class TestInsetFiltergraph:
    def test_graph_places_screen_and_camera(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        assert "split=3" in graph
        assert "[screen]" in graph and "[cam]" in graph
        assert graph.endswith("[v]")

    def test_screen_band_is_never_cropped(self):
        # The whole point: a game HUD or a desktop keeps its edges.
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        screen = graph.split("[sa]")[-1].split("[screen]")[0]
        assert "crop" not in screen
        assert "scale=1080:608" in screen

    def test_camera_band_keeps_the_inset_aspect(self):
        # A 16:9 inset must not be stretched into a wider band; that shows up
        # instantly on a face.
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 356, 200))
        cam = graph.split("[ca]")[-1].split("[cam]")[0]
        crop_w = int(cam.split("w=")[1].split(":")[0])
        crop_h = int(cam.split("h=")[1].split(":")[0])
        band_h = int(cam.split("scale=1080:")[1].split("[")[0])
        assert abs((crop_w / crop_h) - (1080 / band_h)) < 0.05

    def test_camera_band_is_capped(self):
        # A near-square inset would otherwise claim most of the frame.
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 700, 300, 290))
        band_h = int(graph.split("[cam]")[0].split("scale=1080:")[-1])
        assert band_h <= 1920 * 0.40 + 2

    def test_crop_coordinates_stay_inside_the_source(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1850, 1000, 70, 80))
        cam = graph.split("[ca]")[-1].split("[cam]")[0]
        x = int(cam.split("x=")[1].split(":")[0])
        w = int(cam.split("w=")[1].split(":")[0])
        assert x >= 0 and x + w <= 1920

    def test_every_dimension_is_even(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1501, 881, 351, 201))
        cam = graph.split("[ca]")[-1].split("[cam]")[0]
        for key in ("w=", "h=", "x=", "y="):
            assert int(cam.split(key)[1].split(":")[0].split(",")[0]) % 2 == 0


def _crop(graph, label):
    """(w, h, x, y) of the crop on the chain that starts at ``label``."""
    chain = graph.split(f"[{label}]crop=")[1].split(",")[0]
    vals = dict(kv.split("=") for kv in chain.split(":"))
    return int(vals["w"]), int(vals["h"]), int(vals["x"]), int(vals["y"])


class TestGameplayCrop:
    BAND = 1080 / 960  # 50/50 split of a 1080x1920 frame

    def test_centre_crop_when_the_webcam_is_clear_of_it(self):
        # A webcam hard in the right corner sits outside the centre 1214px.
        x, y, w, h = gameplay_crop(1920, 1080, self.BAND, (1650, 750, 270, 330))
        assert (w, h, y) == (1214, 1080, 0)
        assert x == 352  # (1920 - 1214) // 2, rounded down to even

    def test_steps_left_off_a_webcam_on_the_right(self):
        cam = (1400, 700, 300, 300)
        x, _y, w, _h = gameplay_crop(1920, 1080, self.BAND, cam)
        assert x + w <= cam[0]
        assert x >= 0

    def test_steps_right_off_a_webcam_on_the_left(self):
        cam = (200, 700, 300, 300)
        x, _y, w, _h = gameplay_crop(1920, 1080, self.BAND, cam)
        assert x >= cam[0] + cam[2]
        assert x + w <= 1920

    def test_square_source_is_trimmed_not_squashed(self):
        x, y, w, h = gameplay_crop(1080, 1080, self.BAND, (900, 800, 160, 200))
        assert w <= 1080 and abs(w / h - self.BAND) < 0.01
        assert 0 <= y and y + h <= 1080 and y % 2 == 0

    def test_never_rounds_into_the_webcam(self):
        # Odd edges on both sides: the even rounding must step away, not in.
        right = (1401, 700, 301, 300)
        x, _y, w, _h = gameplay_crop(1920, 1080, self.BAND, right)
        assert x + w <= right[0]
        left = (201, 700, 300, 300)
        x, _y, _w, _h = gameplay_crop(1920, 1080, self.BAND, left)
        assert x >= left[0] + left[2]

    def test_stays_inside_the_frame_when_it_cannot_step_around(self):
        # Too wide to avoid: keep the crop in frame rather than fail.
        x, _y, w, _h = gameplay_crop(1920, 1080, self.BAND, (700, 0, 900, 1080))
        assert 0 <= x and x + w <= 1920 and x % 2 == 0


class TestStreamerFiltergraph:
    CAM = (1500, 700, 380, 300)

    def test_webcam_on_top_gameplay_below_filling_the_frame(self):
        graph = streamer_filtergraph(1920, 1080, 1080, 1920, self.CAM)
        assert "[cam][game]vstack=inputs=2" in graph
        cam_h = int(graph.split("[ca]crop")[1].split("scale=1080:")[1].split("[")[0])
        game_h = int(graph.split("[ga]crop")[1].split("scale=1080:")[1].split("[")[0])
        assert cam_h + game_h == 1920
        assert cam_h == int(1920 * STREAMER_CAMERA_RATIO)
        assert graph.endswith("[v]")

    def test_neither_band_is_stretched(self):
        graph = streamer_filtergraph(1920, 1080, 1080, 1920, self.CAM)
        for label, band_h in (("ca", 960), ("ga", 960)):
            w, h, _x, _y = _crop(graph, label)
            assert abs(w / h - 1080 / band_h) < 0.01, label

    def test_webcam_band_covers_the_webcam(self):
        w, h, x, y = _crop(streamer_filtergraph(1920, 1080, 1080, 1920, self.CAM), "ca")
        bx, by, bw, bh = self.CAM
        assert x <= bx + bw / 2 <= x + w and y <= by + bh / 2 <= y + h

    def test_the_gpu_render_reads_it(self):
        # CLAUDE.md: a graph layers_from_graph cannot read silently renders
        # through ffmpeg instead of the one-pass GPU render.
        graph = streamer_filtergraph(1920, 1080, 1080, 1920, self.CAM)
        layers = gpu_render.layers_from_graph(graph, 1920, 1080, 1080, 1920)
        assert len(layers) == 2
        assert layers[0]["dst"][1] == 0 and layers[1]["dst"][1] == 960


class TestStreamMode:
    def _mode(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("STREAM_LAYOUT", raising=False)
        else:
            monkeypatch.setenv("STREAM_LAYOUT", value)
        return importlib.reload(camera_inset).STREAM_MODE

    def test_reads_the_two_modes(self, monkeypatch):
        try:
            assert self._mode(monkeypatch, "general") == "general"
            assert self._mode(monkeypatch, " Streamer ") == "streamer"
        finally:
            self._mode(monkeypatch, None)

    def test_off_by_default_and_for_unknown_values(self, monkeypatch):
        try:
            assert self._mode(monkeypatch, None) == ""
            assert self._mode(monkeypatch, "podcast") == ""
        finally:
            self._mode(monkeypatch, None)
