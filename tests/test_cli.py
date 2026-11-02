import pytest

from gaze_track import cli


@pytest.mark.parametrize(
    "argv,fn,camera",
    [
        (["run", "--camera", "0"], "cmd_run", 0),
        (["--camera", "1", "run"], "cmd_run", 1),
        (["calibrate", "--camera", "0", "--profile", "me"], "cmd_calibrate", 0),
        (["bench"], "cmd_bench", None),
    ],
)
def test_camera_flag_before_or_after_subcommand(monkeypatch, argv, fn, camera):
    got = {}
    for name in ("cmd_run", "cmd_calibrate", "cmd_bench", "cmd_cameras"):
        monkeypatch.setattr(cli, name, lambda a, n=name: got.update(fn=n, camera=a.camera))
    cli.main(argv)
    assert got == {"fn": fn, "camera": camera}
