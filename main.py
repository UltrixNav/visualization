from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


"""TDoA3 UDP Visualizer

Listens for JSON UDP packets from `tdoa3_solver/main.py` and visualizes:
  - Anchor (base-station) positions
  - Latest solved tag position
  - Short trajectory tail
  - Optional: TDoA hyperbolas from meas[] (level set of ||p-Ar||-||p-Aref|| = dd)

Packet example (received as UTF-8 JSON):
{
  "t": 1737.123,
  "ref": 3,
  "src": 3,
  "seq": 12,
  "remote_count": 2,
  "anchors": {"1": [x,y,z], "2": [x,y,z]},
  "meas": [{"id":2,"dd":1.234}, {"id":1,"dd":0.456}],
  "pos": [1.7, 1.6, 1.87],
  "ok": true,
  "rms": 0.0,
  "it": 4,
  "clockCorr": 1.0000012
}
"""


# -----------------------------
# Settings
# -----------------------------
UDP_PORT = 7777
UDP_BIND = "0.0.0.0"  # listen on all interfaces

PLOT_FIXED_Z: float | None = None  # If None use ref.z from anchors
SHOW_HYPERBOLAS = False
GRID_STEP_M = 0.05
MARGIN_M = 0.5
MAX_TRAJ_POINTS = 300

# Kalman filter (constant-velocity in 2D)
USE_KALMAN = True

# Measurement noise std-dev (meters). Increase if you want more smoothing.
KALMAN_R_STD_M = 0.5

# Process noise (how much you allow acceleration / model mismatch). Increase to track faster motion.
KALMAN_Q_ACCEL_STD_MPS2 = 5

# Outlier gate: if a measurement is too inconsistent with prediction, ignore it.
# This helps with rare 1m “teleports”. Typical good range: 3..8.
KALMAN_GATE_NSIGMA = 3.0

# If True, show raw points in addition to the filtered estimate.
SHOW_RAW_POS = False


@dataclass
class State:
	anchors: Dict[int, np.ndarray] | None = None
	last_packet: dict | None = None
	traj_xy: list[tuple[float, float]] = None  # type: ignore
	traj_xy_raw: list[tuple[float, float]] = None  # type: ignore
	kf: "Kalman2D" | None = None

	def __post_init__(self) -> None:
		if self.traj_xy is None:
			self.traj_xy = []
		if self.traj_xy_raw is None:
			self.traj_xy_raw = []
		if self.kf is None:
			self.kf = Kalman2D()


class Kalman2D:
	"""Constant-velocity Kalman filter for 2D position.

	State: [x, y, vx, vy]
	Meas:  [x, y]
	"""

	def __init__(self) -> None:
		self.x: np.ndarray | None = None
		self.P: np.ndarray | None = None
		self.t_last: float | None = None

	def reset(self) -> None:
		self.x = None
		self.P = None
		self.t_last = None

	def step(self, z_xy: np.ndarray, t: float) -> np.ndarray:
		"""Update filter with a new position measurement at time t (seconds)."""
		z_xy = np.asarray(z_xy, dtype=float).reshape(2)

		if self.x is None or self.P is None or self.t_last is None:
			# Initialize: assume zero velocity, fairly uncertain.
			self.x = np.array([z_xy[0], z_xy[1], 0.0, 0.0], dtype=float)
			self.P = np.diag([1.0, 1.0, 4.0, 4.0]).astype(float)
			self.t_last = float(t)
			return self.x[:2].copy()

		dt = float(t - self.t_last)
		# Be robust to time glitches / missing timestamps.
		if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
			dt = 1.0 / 30.0

		# Predict
		F = np.array(
			[
				[1.0, 0.0, dt, 0.0],
				[0.0, 1.0, 0.0, dt],
				[0.0, 0.0, 1.0, 0.0],
				[0.0, 0.0, 0.0, 1.0],
			],
			dtype=float,
		)

		# Constant-acceleration process noise projected into [x,y,vx,vy].
		# Q = sigma_a^2 * [[dt^4/4, 0, dt^3/2, 0], ...]
		sa2 = float(KALMAN_Q_ACCEL_STD_MPS2) ** 2
		dt2 = dt * dt
		dt3 = dt2 * dt
		dt4 = dt2 * dt2
		q11 = dt4 / 4.0
		q13 = dt3 / 2.0
		q33 = dt2
		Q = sa2 * np.array(
			[
				[q11, 0.0, q13, 0.0],
				[0.0, q11, 0.0, q13],
				[q13, 0.0, q33, 0.0],
				[0.0, q13, 0.0, q33],
			],
			dtype=float,
		)

		x_pred = F @ self.x
		P_pred = F @ self.P @ F.T + Q

		# Update
		H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
		r = float(KALMAN_R_STD_M) ** 2
		R = np.diag([r, r]).astype(float)

		y = z_xy - (H @ x_pred)
		S = H @ P_pred @ H.T + R
		Sinv = np.linalg.inv(S)

		# Innovation gating (Mahalanobis distance)
		m2 = float(y.T @ Sinv @ y)
		gate2 = float(KALMAN_GATE_NSIGMA) ** 2
		if np.isfinite(m2) and m2 > gate2:
			# Skip update (treat as outlier), but keep predicted state.
			self.x = x_pred
			self.P = P_pred
			self.t_last = float(t)
			return self.x[:2].copy()

		K = P_pred @ H.T @ Sinv
		x_upd = x_pred + (K @ y)
		P_upd = (np.eye(4) - K @ H) @ P_pred

		self.x = x_upd
		self.P = P_upd
		self.t_last = float(t)
		return self.x[:2].copy()


def _parse_anchors(raw: Any) -> Dict[int, np.ndarray] | None:
	if not isinstance(raw, dict):
		return None

	out: Dict[int, np.ndarray] = {}
	for k, v in raw.items():
		try:
			aid = int(k)
			arr = np.asarray(v, dtype=float)
			if arr.shape != (3,):
				continue
			out[aid] = arr
		except Exception:
			continue
	return out or None


def _build_grid(anchors: Dict[int, np.ndarray], margin_m: float, step_m: float):
	pts = np.stack(list(anchors.values()), axis=0)
	min_xy = pts[:, :2].min(axis=0) - margin_m
	max_xy = pts[:, :2].max(axis=0) + margin_m
	x = np.arange(float(min_xy[0]), float(max_xy[0]) + step_m, step_m)
	y = np.arange(float(min_xy[1]), float(max_xy[1]) + step_m, step_m)
	X, Y = np.meshgrid(x, y)
	lims = (float(min_xy[0]), float(max_xy[0]), float(min_xy[1]), float(max_xy[1]))
	return X, Y, lims


def _draw(ax, state: State, grid_cache) -> None:
	ax.clear()

	pkt = state.last_packet
	anchors = state.anchors
	if pkt is None or anchors is None:
		ax.set_title("Waiting for UDP packets...")
		ax.grid(True, alpha=0.25)
		return

	ref_id = int(pkt.get("ref", -1))
	ok = bool(pkt.get("ok", False))
	rms = pkt.get("rms", None)
	seq = pkt.get("seq", None)
	pos = pkt.get("pos", None)
	pt = pkt.get("t", None)
	meas = pkt.get("meas", [])

	# Anchors
	for aid, p in anchors.items():
		ax.scatter([p[0]], [p[1]], c="tab:red" if aid == ref_id else "k", s=55, zorder=5)
		ax.text(p[0] + 0.03, p[1] + 0.03, f"A{aid}", fontsize=9)

	if ref_id not in anchors:
		ax.set_title("ref anchor missing in packet")
		ax.grid(True, alpha=0.25)
		return

	X, Y, lims = grid_cache
	ref = anchors[ref_id]
	z_use = float(PLOT_FIXED_Z) if PLOT_FIXED_Z is not None else float(ref[2])

	# Hyperbolas
	if SHOW_HYPERBOLAS and isinstance(meas, list):
		Z = np.full_like(X, z_use)
		Rref = np.sqrt((X - ref[0]) ** 2 + (Y - ref[1]) ** 2 + (Z - ref[2]) ** 2)

		for m in meas:
			if not isinstance(m, dict):
				continue
			try:
				rid = int(m.get("id"))
				dd = float(m.get("dd"))
			except Exception:
				continue
			if rid not in anchors:
				continue
			ar = anchors[rid]
			Rr = np.sqrt((X - ar[0]) ** 2 + (Y - ar[1]) ** 2 + (Z - ar[2]) ** 2)
			F = (Rr - Rref) - dd
			ax.contour(X, Y, F, levels=[0.0], linewidths=1.3, alpha=0.75)

	# Position + trajectory
	if isinstance(pos, list) and len(pos) >= 2:
		try:
			x_raw, y_raw = float(pos[0]), float(pos[1])
			if SHOW_RAW_POS:
				state.traj_xy_raw.append((x_raw, y_raw))
				if len(state.traj_xy_raw) > MAX_TRAJ_POINTS:
					state.traj_xy_raw = state.traj_xy_raw[-MAX_TRAJ_POINTS:]

			# Use solver timestamp if present, otherwise wall-clock.
			try:
				t_meas = float(pt) if pt is not None else time.time()
			except Exception:
				t_meas = time.time()

			if USE_KALMAN and state.kf is not None:
				xy = state.kf.step(np.array([x_raw, y_raw], dtype=float), t_meas)
				x, y = float(xy[0]), float(xy[1])
			else:
				x, y = x_raw, y_raw

			state.traj_xy.append((x, y))
			if len(state.traj_xy) > MAX_TRAJ_POINTS:
				state.traj_xy = state.traj_xy[-MAX_TRAJ_POINTS:]
			tx = [p[0] for p in state.traj_xy]
			ty = [p[1] for p in state.traj_xy]
			ax.plot(tx, ty, color="tab:blue", linewidth=1.2, alpha=0.6, zorder=3)
			ax.scatter([x], [y], c="tab:blue" if ok else "tab:orange", marker="x", s=50, zorder=6)

			if SHOW_RAW_POS:
				rx = [p[0] for p in state.traj_xy_raw]
				ry_ = [p[1] for p in state.traj_xy_raw]
				ax.plot(rx, ry_, color="0.3", linewidth=1.0, alpha=0.25, zorder=2)
				ax.scatter([x_raw], [y_raw], c="0.2", marker=".", s=20, alpha=0.35, zorder=4)
		except Exception:
			pass

	title = f"ref=A{ref_id} seq={seq} ok={ok}"
	if rms is not None:
		try:
			title += f" rms={float(rms):.3f}"
		except Exception:
			pass
	ax.set_title(title)
	ax.set_xlabel("x [m]")
	ax.set_ylabel("y [m]")
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(lims[0], lims[1])
	ax.set_ylim(lims[2], lims[3])
	ax.grid(True, alpha=0.25)


def main() -> int:
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((UDP_BIND, UDP_PORT))
	sock.setblocking(False)

	state = State()
	fig, ax = plt.subplots(1, 1, figsize=(7, 6))
	plt.ion()
	plt.show(block=False)

	grid_cache = None

	last_draw = 0.0
	draw_hz = 30.0

	while True:
		# receive as many packets as available
		for _ in range(50):
			try:
				data, _addr = sock.recvfrom(65535)
			except BlockingIOError:
				break
			except OSError:
				break

			try:
				pkt = json.loads(data.decode("utf-8", errors="replace"))
			except Exception:
				continue

			if not isinstance(pkt, dict):
				continue

			state.last_packet = pkt
			anchors = _parse_anchors(pkt.get("anchors"))
			if anchors is not None:
				# If anchors changed or first time, rebuild grid.
				if state.anchors is None or set(state.anchors.keys()) != set(anchors.keys()):
					state.traj_xy = []
					state.traj_xy_raw = []
					if state.kf is not None:
						state.kf.reset()
					state.anchors = anchors
					grid_cache = _build_grid(state.anchors, MARGIN_M, GRID_STEP_M)
				else:
					state.anchors = anchors

		now = time.time()
		if grid_cache is not None and (now - last_draw) >= 1.0 / draw_hz:
			last_draw = now
			_draw(ax, state, grid_cache)
			ax.figure.canvas.draw_idle()
			ax.figure.canvas.flush_events()

		time.sleep(0.005)


if __name__ == "__main__":
	raise SystemExit(main())