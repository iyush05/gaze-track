"""Review report: one self-contained HTML file per recorded session for a human reviewer.

Flags are signals for review, never verdicts. The report shows *what* was flagged, *when*, how
reliable tracking was at the time, and *what the camera saw* (snapshots at the start, middle and end
of each flagged episode), so a person can judge each flag in seconds.
"""

from __future__ import annotations

import base64
import html
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .data.session import load_session
from .evaluate import _replay, _replay_setup

ZONE_COLORS = {
    "ON_SCREEN": "#3f9d5a", "EDGE": "#8fbf5a", "DOWN_KEYBOARD": "#5b8fd6", "NEAR_OFF": "#e3a33b",
    "FAR_OFF": "#d9534f", "EYES_CLOSED": "#9a8fbf", "NO_FACE": "#6b6b6b", "MULTI_FACE": "#b03fb0",
    "UNKNOWN": "#c9c9c9",
}
ZONE_TEXT = {
    "ON_SCREEN": "on screen", "EDGE": "screen edge", "DOWN_KEYBOARD": "keyboard", "NEAR_OFF": "just off screen",
    "FAR_OFF": "away from screen", "EYES_CLOSED": "eyes closed", "NO_FACE": "no face", "MULTI_FACE": "several faces",
    "UNKNOWN": "not judged (image quality)",
}
KIND_TEXT = {
    "sustained_off_screen": "Looked away from the screen",
    "repeated_directional_glance": "Repeated glances to the same spot",
    "head_turned_away": "Head turned to the side",
    "repeated_head_turns": "Repeated head turns",
    "face_absent": "Face not visible",
    "multiple_faces": "More than one face in view",
    "eyes_closed_long": "Eyes closed for a long time",
    "tracking_unreliable": "Tracking unreliable (image too dark / blurry)",
}
SEVERITY_COLORS = {"warning": "#d98a1f", "alert": "#c9302c"}


def describe_direction(yaw: float | None, pitch: float | None) -> str:
    """Gaze/head direction in plain words, from the user's point of view."""
    if yaw is None and pitch is None:
        return ""
    parts = []
    if pitch is not None and pitch < -8:
        parts.append("up")
    elif pitch is not None and pitch > 8:
        parts.append("down")
    if yaw is not None and yaw > 8:
        parts.append("left")
    elif yaw is not None and yaw < -8:
        parts.append("right")
    return "-".join(parts) if parts else "straight ahead"


INCIDENT_GAP_S = 10.0  # flags closer than this belong to one incident (one card to review)
INCIDENT_MAX_S = 60.0  # ...but an incident never spans more than this (keeps evidence reviewable)


def group_incidents(flags: list, gap_s: float = INCIDENT_GAP_S, max_s: float = INCIDENT_MAX_S,
                    breaks: list[float] | None = None) -> list[list]:
    """Group flags into incidents: a new incident starts when the next flag begins more than `gap_s`
    after the previous flag, when the incident would exceed `max_s`, or across a recording break
    (times in `breaks`, e.g. pauses between scripted segments)."""
    breaks = sorted(breaks or [])
    out: list[list] = []
    for e in sorted(flags, key=lambda e: e.t):
        start = e.t - (e.duration_s or 0.0)
        if out:
            inc = out[-1]
            inc_start = min(x.t - (x.duration_s or 0.0) for x in inc)
            crosses = any(inc[-1].t < b <= e.t for b in breaks)
            if start - inc[-1].t <= gap_s and e.t - inc_start <= max_s and not crosses:
                inc.append(e)
                continue
        out.append([e])
    return out


def _thumb(path: Path, width: int = 240) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return ""
    img = cv2.flip(img, 1)  # mirrored like a selfie, so "left" in the text matches the picture
    h = int(img.shape[0] * width / img.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _fmt_t(sec: float) -> str:
    m, s = divmod(max(0.0, sec), 60)
    return f"{int(m):02d}:{s:04.1f}"


def _timeline_svg(frames: list[dict], events: list, t0: float, dur: float, truth: list[tuple] | None,
                  anchor=lambda k, e: f"ev{k}") -> str:
    W, row_h = 1000, 26
    rows_y = {"zone": 22}
    height = 22 + row_h + 20
    if truth:
        rows_y["truth"] = 22 + row_h + 14
        height += row_h + 14
    sx = lambda t: (t - t0) / max(dur, 1e-6) * W  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {height}" class="timeline" role="img" aria-label="zone timeline">']
    # Zone strip: merge consecutive frames with the same zone.
    start = None
    for i, f in enumerate(frames):
        if start is None:
            start = i
        last = i == len(frames) - 1
        if last or frames[i + 1]["zone"] != f["zone"] or frames[i + 1]["t"] - f["t"] > 2.0:
            a, b = frames[start]["t"], (f["t"] + (1 / 15))
            z = f["zone"]
            parts.append(f'<rect x="{sx(a):.1f}" y="{rows_y["zone"]}" width="{max(0.6, sx(b) - sx(a)):.1f}" '
                         f'height="{row_h}" fill="{ZONE_COLORS.get(z, "#999")}"><title>{_fmt_t(a - t0)}–'
                         f'{_fmt_t(b - t0)} {html.escape(ZONE_TEXT.get(z, z))}</title></rect>')
            start = None
    if truth:
        for a, b, label, task in truth:
            col = "#c9302c" if label == "cheat" else "#3f9d5a"
            parts.append(f'<rect x="{sx(a):.1f}" y="{rows_y["truth"]}" width="{max(0.6, sx(b) - sx(a)):.1f}" '
                         f'height="{row_h}" fill="{col}" opacity="0.85"><title>{html.escape(task)}</title></rect>')
            parts.append(f'<text x="{sx(a) + 4:.1f}" y="{rows_y["truth"] + 17}" class="tl-label">'
                         f'{html.escape(task.replace("_", " "))}</text>')
    for k, e in enumerate(events):
        x = sx(e.t)
        col = SEVERITY_COLORS.get(e.severity, "#888")
        parts.append(f'<a href="#{anchor(k, e)}"><path d="M{x - 6:.1f},2 L{x + 6:.1f},2 L{x:.1f},16 Z" fill="{col}">'
                     f'<title>{_fmt_t(e.t - t0)} {html.escape(KIND_TEXT.get(e.kind, e.kind))}</title></path></a>')
    parts.append(f'<text x="0" y="{height - 4}" class="tl-axis">00:00</text>'
                 f'<text x="{W}" y="{height - 4}" class="tl-axis" text-anchor="end">{_fmt_t(dur)}</text></svg>')
    return "".join(parts)


def _render(*, name: str, subtitle: str, estimator: str, frames: list[dict], flags: list, truth,
            cal_stats: dict, snapshots, max_events: int = 60) -> str:
    """HTML for a review report. frames: [{t, zone, quality}], flags: warning/alert Events,
    snapshots(incident_flags, start_t, end_t) → [(t, zone, data_url)] (≤ 4)."""
    zones = Counter(f["zone"] for f in frames)
    t0 = frames[0]["t"]
    dur = max(frames[-1]["t"] - t0, 1e-6)
    n = max(1, sum(zones.values()))
    zone_rows = "".join(
        f'<tr><td><span class="sw" style="background:{ZONE_COLORS.get(z, "#999")}"></span>'
        f'{html.escape(ZONE_TEXT.get(z, z))}</td><td class="num">{100 * c / n:.1f}%</td></tr>'
        for z, c in zones.most_common())
    qual = Counter(f["quality"] for f in frames)
    unknown = 100 * zones.get("UNKNOWN", 0) / n
    kinds = Counter(e.kind for e in flags)
    sev = Counter(e.severity for e in flags)
    cal = cal_stats

    breaks = [frames[i + 1]["t"] for i in range(len(frames) - 1) if frames[i + 1]["t"] - frames[i]["t"] > 2.0]
    incidents = group_incidents(flags, breaks=breaks)
    flag_incident = {id(e): k for k, inc in enumerate(incidents) for e in inc}
    cards = []
    for k, inc in enumerate(incidents[:max_events]):
        sev_max = "alert" if any(e.severity == "alert" for e in inc) else "warning"
        a = min(e.t - (e.duration_s or 0.0) for e in inc)
        b = max(e.t for e in inc)
        imgs = "".join(
            f'<figure><img src="{src}" alt="camera frame at {_fmt_t(t - t0)}" '
            f'loading="lazy"><figcaption>{_fmt_t(t - t0)} · {html.escape(ZONE_TEXT.get(z, z))}'
            f'</figcaption></figure>' for t, z, src in snapshots(inc, a, b))
        items = []
        for e in inc:
            bits = [_fmt_t(e.t - t0), html.escape(KIND_TEXT.get(e.kind, e.kind))]
            if e.duration_s:
                bits.append(f"{e.duration_s:.1f} s")
            d = describe_direction(e.yaw_deg, e.pitch_deg)
            if d:
                bits.append(d)
            if e.detail:
                bits.append(html.escape(e.detail))
            items.append(f'<li><span class="dot {e.severity}"></span>{" · ".join(bits)}</li>')
        kinds_here = list(dict.fromkeys(KIND_TEXT.get(e.kind, e.kind) for e in inc))
        seg = ""
        if truth:
            segs = [(label, task) for ta, tb, label, task in truth if ta - 0.5 <= b and a <= tb + 0.5]
            seg = "".join(f'<span class="seg {label}">recorded segment: {html.escape(task)}</span>'
                          for label, task in dict.fromkeys(segs))
        cards.append(
            f'<article class="card" id="inc{k}" data-incident="{k}"><header><span class="badge {sev_max}">{sev_max}</span>'
            f'<h3>Incident {k + 1}: {html.escape(" + ".join(kinds_here))}</h3>{seg}</header>'
            f'<p class="meta">{_fmt_t(a - t0)}–{_fmt_t(b - t0)} · {len(inc)} flag{"s" if len(inc) > 1 else ""}</p>'
            f'<ul class="flags">{"".join(items)}</ul><div class="shots">{imgs}</div>'
            f'<div class="decide" role="group" aria-label="reviewer decision">'
            + "".join(f'<label><input type="radio" name="d{k}" value="{v}"> {v}</label>'
                      for v in ("confirmed", "dismissed", "unsure"))
            + f'<input type="text" class="note" placeholder="note (optional)" aria-label="note for incident {k + 1}">'
            f'</div></article>')
    more = (f'<p class="muted">{len(incidents) - max_events} further incidents not shown.</p>'
            if len(incidents) > max_events else "")
    no_flags = '<p class="ok-note">No warnings or alerts in this session.</p>' if not flags else ""

    title = f"Review: {name}"
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f7f7f5; --card:#ffffff; --text:#1d1d1b; --muted:#6a6a66; --line:#e2e1dc; --accent:#2d6cdf; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --card:#21211f; --text:#ecebe6; --muted:#a3a29c; --line:#34332f; --accent:#7aa7ff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 -apple-system, system-ui, "Segoe UI", sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:28px 0 10px; }} h3 {{ font-size:15px; margin:0; }}
.muted, .meta {{ color:var(--muted); }} .meta {{ margin:6px 0 10px; font-size:14px; }}
.notice {{ border-left:4px solid var(--accent); background:var(--card); padding:10px 14px; margin:14px 0; border-radius:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.stat b {{ display:block; font-size:22px; font-variant-numeric:tabular-nums; }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
table {{ border-collapse:collapse; width:100%; }} td {{ padding:3px 6px; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.sw {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:8px; vertical-align:-1px; }}
.timeline {{ width:100%; height:auto; }} .tl-label {{ font-size:11px; fill:#fff; }} .tl-axis {{ font-size:11px; fill:var(--muted); }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size:13px; color:var(--muted); margin-top:8px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:12px 0; }}
.card header {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; }}
.badge {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:#fff; padding:2px 8px; border-radius:999px; }}
.badge.warning {{ background:#b8741a; }} .badge.alert {{ background:#c9302c; }}
.seg {{ font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); }}
.seg.cheat {{ color:#c9302c; }} .seg.honest {{ color:#3f9d5a; }}
.shots {{ display:flex; flex-wrap:wrap; gap:8px; }}
figure {{ margin:0; }} figure img {{ width:240px; max-width:100%; border-radius:6px; display:block; }}
figcaption {{ font-size:12px; color:var(--muted); margin-top:2px; }}
.flags {{ margin:4px 0 10px; padding-left:0; list-style:none; font-size:14px; }}
.flags li {{ margin:2px 0; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; vertical-align:1px; }}
.dot.warning {{ background:#d98a1f; }} .dot.alert {{ background:#c9302c; }}
.decide {{ display:flex; flex-wrap:wrap; align-items:center; gap:6px 14px; margin-top:10px; padding-top:10px;
  border-top:1px solid var(--line); font-size:14px; }}
.decide .note {{ flex:1 1 220px; min-width:0; padding:5px 8px; border:1px solid var(--line); border-radius:6px;
  background:var(--bg); color:var(--text); }}
button {{ font:inherit; padding:7px 12px; border-radius:6px; border:1px solid var(--line); background:var(--card);
  color:var(--text); cursor:pointer; }}
.ok-note {{ background:var(--card); border:1px solid var(--line); padding:12px 14px; border-radius:8px; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="muted">{html.escape(subtitle)} · {_fmt_t(dur)} reviewed · estimator: {html.escape(estimator)} ·
report generated {time.strftime("%Y-%m-%d %H:%M")}</p>
<div class="notice">These are <b>signals for human review, not verdicts.</b> Each flag shows what the camera saw; please
judge every flag yourself. Natural behaviour (thinking, typing, brief glances) is expected and is not flagged by design.</div>

<div class="grid">
<div class="stat"><span class="muted">Incidents to review</span><b>{len(incidents)}</b></div>
<div class="stat"><span class="muted">Flags (alerts / warnings)</span><b>{sev.get("alert", 0)} / {sev.get("warning", 0)}</b></div>
<div class="stat"><span class="muted">Not judged (image quality)</span><b>{unknown:.1f}%</b></div>
<div class="stat"><span class="muted">Calibration error</span><b>{cal.get("lopo_mean_err_deg", float("nan")):.1f}°</b></div>
</div>

<h2>Timeline</h2>
<div class="panel">{_timeline_svg(frames, flags, t0, dur, truth, anchor=lambda k, e: f"inc{flag_incident[id(e)]}")}
<div class="legend">{"".join(f'<span><span class="sw" style="background:{c}"></span>{html.escape(ZONE_TEXT[z])}</span>' for z, c in ZONE_COLORS.items() if zones.get(z))}
<span>▼ flag (orange: warning, red: alert)</span>{'<span>bottom row: recorded segments (green honest, red cheating)</span>' if truth else ''}</div></div>

<h2>Summary</h2>
<div class="grid">
<div class="panel"><b>Time in each zone</b><table>{zone_rows}</table></div>
<div class="panel"><b>Flags by type</b><table>{"".join(f'<tr><td>{html.escape(KIND_TEXT.get(k, k))}</td><td class="num">{v}</td></tr>' for k, v in kinds.most_common()) or '<tr><td class="muted">none</td></tr>'}</table></div>
<div class="panel"><b>Image conditions</b><table>{"".join(f'<tr><td>{html.escape(k)}</td><td class="num">{100 * v / max(1, len(frames)):.1f}%</td></tr>' for k, v in qual.most_common())}</table></div>
</div>

<h2>Incidents ({len(incidents)})</h2>
<p class="muted">Flags less than {INCIDENT_GAP_S:.0f} s apart are grouped into one incident. Mark each one; decisions are
kept in this browser and can be downloaded.</p>
{no_flags}{"".join(cards)}{more}
<p><button id="dl" type="button">Download decisions (JSON)</button> <span id="saved" class="muted"></span></p>
<p class="muted">Snapshots are mirrored (like a selfie), so "left" in the text is on the left of the picture.
Generated locally by gaze-track; nothing was uploaded.</p>
</main>
<script>
(() => {{
  const KEY = "gaze-track-review:" + {json.dumps(name)};
  let state = {{}};
  try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ state = {{}}; }}
  const save = () => {{
    try {{ localStorage.setItem(KEY, JSON.stringify(state)); document.getElementById("saved").textContent = "saved in this browser"; }}
    catch (e) {{ document.getElementById("saved").textContent = "could not save in this browser; use Download"; }}
  }};
  document.querySelectorAll("[data-incident]").forEach(card => {{
    const k = card.dataset.incident, st = state[k] || {{}};
    card.querySelectorAll("input[type=radio]").forEach(r => {{
      r.checked = st.decision === r.value;
      r.addEventListener("change", () => {{ state[k] = {{...(state[k] || {{}}), decision: r.value}}; save(); }});
    }});
    const note = card.querySelector(".note");
    note.value = st.note || "";
    note.addEventListener("input", () => {{ state[k] = {{...(state[k] || {{}}), note: note.value}}; save(); }});
  }});
  document.getElementById("dl").addEventListener("click", () => {{
    const data = {{session: {json.dumps(name)}, reviewed_at: new Date().toISOString(), incidents: {json.dumps(
        [{"incident": k + 1, "start_s": round(min(e.t - (e.duration_s or 0.0) for e in inc) - t0, 1),
          "end_s": round(max(e.t for e in inc) - t0, 1), "flags": [e.kind for e in inc]} for k, inc in enumerate(incidents)])}
      .map((inc, i) => ({{...inc, ...(state[i] || {{}})}}))}};
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {{type: "application/json"}}));
    a.download = {json.dumps(name + "_review.json")};
    a.click();
  }});
}})();
</script>
</body></html>"""
    return doc


def build_report(session_path: str | Path, out: str | Path | None = None, model: str | None = None,
                 base: Config | None = None, max_events: int = 60, profile: str | None = None) -> Path:
    """profile: calibration profile name/path; default = calibrate from the session's own dots."""
    base = base or Config()
    s = load_session(session_path)
    setup = _replay_setup(s, base, profile, True, model, bool(model))
    rows = [r for r in s.rows if r["segment"] != "calib"] or list(s.rows)
    frames: list[dict] = []

    def on_frame(r, res):
        q = getattr(res, "quality", None)
        frames.append({"t": float(r["t"]), "frame": int(r["frame"]), "zone": res.zone.value,
                       "quality": q.status if q is not None else "ok"})

    events, _ = _replay(s, rows, *setup, on_frame=on_frame, quality=True)
    flags = [e for e in events if e.severity != "info"]
    ts = np.array([f["t"] for f in frames])

    def frame_at(t: float) -> dict:
        return frames[int(np.clip(np.searchsorted(ts, t), 0, len(frames) - 1))]

    # Ground truth (scripted behaviour / guided sessions), to let a reviewer check the system.
    truth = None
    if s.meta.get("kind") == "behavior":
        truth = []
        for tid in sorted({int(r["trial_id"]) for r in rows if r["segment"] == "behavior"}):
            rr = [r for r in rows if r["segment"] == "behavior" and int(r["trial_id"]) == tid]
            truth.append((float(rr[0]["t"]), float(rr[-1]["t"]), str(rr[0]["zone_gt"]).lower(), rr[0]["task"]))

    def snapshots(inc, a, b):
        # Start of the incident, then each flag's moment (max 4 pictures).
        moments = sorted(set(round(m, 1) for m in [a] + [e.t for e in inc]))
        if len(moments) > 4:
            moments = [moments[int(i)] for i in np.linspace(0, len(moments) - 1, 4)]
        return [(f["t"], f["zone"], _thumb(s.frame_path(f["frame"]))) for f in (frame_at(m) for m in moments)]

    doc = _render(
        name=s.dir.name,
        subtitle=f"Subject {s.subject} · recorded {s.meta.get('created', '')} · glasses {s.meta.get('glasses', '?')}",
        estimator=("fused (" + Path(model).parent.name + ")") if model else "feature mapper (Phase 0)",
        frames=frames, flags=flags, truth=truth, cal_stats=setup[1].stats, snapshots=snapshots,
        max_events=max_events)
    out = Path(out) if out else s.dir / "report.html"
    out.write_text(doc)
    return out


def build_report_from_bundle(bundle: str | Path, out: str | Path | None = None, max_events: int = 60) -> Path:
    """Review report for a live-session evidence bundle (see evidence.py)."""
    from .events import Event
    from .evidence import load_bundle

    bundle = Path(bundle)
    meta, frames, evs = load_bundle(bundle)
    if not frames:
        raise SystemExit(f"{bundle}: empty timeline")
    flags, shots_of = [], {}
    for d in evs:
        e = Event(d["t"], d["kind"], d["severity"], d.get("duration_s", 0.0), d.get("yaw_deg"), d.get("pitch_deg"),
                  d.get("detail", ""))
        flags.append(e)
        shots_of[id(e)] = d.get("snapshots", [])

    def snapshots(inc, a, b):
        seen, out_ = {}, []
        for e in inc:
            for sh in shots_of[id(e)]:
                seen.setdefault(sh["file"], sh)
        items = sorted(seen.values(), key=lambda x: x["t"])
        if len(items) > 4:
            items = [items[int(i)] for i in np.linspace(0, len(items) - 1, 4)]
        for sh in items:
            out_.append((sh["t"], sh["zone"], _thumb(bundle / sh["file"])))
        return out_

    estimator = meta.get("estimator", "?")
    doc = _render(
        name=bundle.name,
        subtitle=f"Live session · profile {meta.get('profile', '?')} · started {meta.get('started', '?')}",
        estimator=estimator, frames=frames, flags=flags, truth=None,
        cal_stats={"lopo_mean_err_deg": meta.get("calibration_err_deg", float("nan"))},
        snapshots=snapshots, max_events=max_events)
    out = Path(out) if out else bundle / "report.html"
    out.write_text(doc)
    return out


def cmd_report(args) -> None:
    base = Config.load(args.config)
    from .cli import resolve_model

    model = resolve_model(args.model) if args.model else None
    from .evidence import is_bundle

    for sp in args.sessions:
        out = (Path(args.out) / f"{Path(sp).name}.html") if args.out else None
        if is_bundle(sp):  # live-session evidence bundle
            p = build_report_from_bundle(sp, out=out)
        else:  # recorded session (replayed)
            p = build_report(sp, model=model, base=base, profile=args.profile, out=out)
        print(f"report → {p}")


def add_parser(sub) -> None:
    c = sub.add_parser("report", help="HTML review report for recorded sessions or live evidence bundles")
    c.add_argument("sessions", nargs="+")
    c.add_argument("--model", default=None, help="learned checkpoint → use the fused estimator")
    c.add_argument("--out", default=None, help="folder for the reports (default: inside each session folder)")
    c.add_argument("--profile", default=None, help="calibration profile instead of the session's own dots")
    c.set_defaults(fn=cmd_report)
