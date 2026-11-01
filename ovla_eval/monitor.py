"""Giám sát thất bại lúc chạy: tín hiệu có sẵn có báo sớm hơn timeout không, khi kiểm soát tỉ lệ báo động sai?

Chỉ cần numpy và các .npz của rollout, không cần GPU hay simulator.

Ngưỡng được hiệu chỉnh kiểu leave-one-task-out: lấy các episode thành công của 9 task để đặt ngưỡng rồi kiểm tra trên
task còn lại. Báo động sai = báo động trên episode thành công. Chạy nhiều tín hiệu cùng lúc thì mỗi tín hiệu nhận
alpha / số tín hiệu, nên tổng báo động sai không vượt alpha (trong phạm vi giả định các task trao đổi được với nhau).

Ví dụ:
    python -m ovla_eval.monitor --rollouts outputs/rollouts/libero_spatial --alarm-signals entropy timer
    python -m ovla_eval.monitor --rollouts outputs/rollouts/libero_object \\
        --frozen outputs/rollouts/libero_spatial/monitor.json
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from .constants import ACTION_DIM_NAMES

SIGNALS = ("entropy", "token_conf", "no_motion", "stall", "timer")
UNION_SIGNALS = ("entropy", "no_motion")  # tín hiệu kết hợp, chốt trước khi xem kết quả
ALPHAS = (0.01, 0.02, 0.05, 0.10)
AUC_STEPS = (30, 50, 70, 90, 110)

DZ_INDEX = ACTION_DIM_NAMES.index("dz")
N_POSE_DIMS = 6  # bỏ gripper: chiều này gần như nhị phân nên không cho biết model phân vân về chuyển động
ROLL_WINDOW = 10
STILL_EPS = 0.001  # m/bước: dịch chuyển dưới mức này thì coi là đứng yên
DZ_DOWN = -0.2  # dz dưới mức này thì coi là đang ra lệnh đi xuống

LIFT_DZ = 0.3  # dz tối thiểu khi nhấc tay; dz >= 0 chỉ làm tay đứng yên, không tạo góc nhìn mới
LIFT_STEPS = 10
MAX_LIFTS = 2  # sau số lần nhấc này mà vẫn báo động thì dừng hẳn
COOLDOWN_STEPS = 10  # chờ sau mỗi lần nhấc, vì entropy trung bình trượt vẫn còn cao ngay sau đó

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 7


class FailureMonitor:
    """Tính điểm từng tín hiệu theo bước, chỉ dùng thông tin đã có tới bước hiện tại. Điểm càng cao càng đáng ngờ."""

    def __init__(self):
        self.t = -1
        self._entropy = deque(maxlen=ROLL_WINDOW)
        self._conf = deque(maxlen=ROLL_WINDOW)
        self._prev_eef = None
        self._no_motion = 0
        self._stall = 0

    def update(self, bin_entropy: np.ndarray, token_prob: np.ndarray, eef_state: np.ndarray, prev_action=None) -> dict:
        """eef_state là trạng thái trước action của bước này, prev_action là action đã thực thi ở bước trước."""
        self.t += 1
        self._entropy.append(float(bin_entropy[:N_POSE_DIMS].mean()))
        self._conf.append(1.0 - float(token_prob[:N_POSE_DIMS].min()))
        if self._prev_eef is not None:
            move = eef_state[:3] - self._prev_eef[:3]
            self._no_motion = self._no_motion + 1 if np.linalg.norm(move) < STILL_EPS else 0
            pressing = prev_action[DZ_INDEX] < DZ_DOWN and abs(move[2]) < STILL_EPS
            self._stall = self._stall + 1 if pressing else 0
        self._prev_eef = eef_state
        full = len(self._entropy) == ROLL_WINDOW  # chưa đủ cửa sổ thì chưa được báo động
        return {
            "entropy": float(np.mean(self._entropy)) if full else -math.inf,
            "token_conf": float(np.mean(self._conf)) if full else -math.inf,
            "no_motion": float(self._no_motion),
            "stall": float(self._stall),
            "timer": float(self.t),
        }


class ResponseGuard:
    """Máy trạng thái cho một episode. Báo động ở bước t thì phản ứng có hiệu lực từ bước t + 1, ở mọi chế độ.

    response:
        hold: giữ nguyên tư thế tới hết episode.
        clip: cắt dz sau khi decode trong LIFT_STEPS bước, rồi chạy tiếp.
        mask: che token dz lúc decode trong LIFT_STEPS bước, rồi chạy tiếp.
    """

    def __init__(self, thresholds: dict, response: str, lift_dz: float = LIFT_DZ, lift_steps: int = LIFT_STEPS):
        self.thresholds = thresholds
        self.response = response
        self.lift_dz = lift_dz
        self.lift_steps = lift_steps
        self.monitor = FailureMonitor()
        self.alarm_steps: list[int] = []
        self._prev_action = None
        self._hold_action = None
        self._lifts = 0
        self._lift_left = 0
        self._cooldown = 0

    @property
    def dz_min(self) -> float | None:
        """Sàn dz cho predict() ở bước sắp tới. None = decode bình thường."""
        return self.lift_dz if self.response == "mask" and self._lift_left > 0 else None

    def step(self, pred, eef_state: np.ndarray) -> np.ndarray:
        """Nhận dự đoán của bước này, trả về action thực sự gửi vào env."""
        action = np.array(pred.env, dtype=np.float64, copy=True)
        if self._hold_action is not None:
            action = self._hold_action
        elif self._lift_left > 0 and self.response == "clip":
            action[DZ_INDEX] = max(action[DZ_INDEX], self.lift_dz)

        scores = self.monitor.update(pred.bin_entropy, pred.token_prob, eef_state, self._prev_action)
        self._prev_action = action
        self._advance(any(scores[s] > thr for s, thr in self.thresholds.items()), action)
        return action

    def _advance(self, alarm: bool, action: np.ndarray) -> None:
        if self._hold_action is not None:
            return
        if self._lift_left > 0:
            self._lift_left -= 1
            if self._lift_left == 0:
                self._cooldown = COOLDOWN_STEPS
        elif self._cooldown > 0:
            self._cooldown -= 1
        elif alarm:
            self.alarm_steps.append(self.monitor.t)
            if self.response == "hold" or self._lifts >= MAX_LIFTS:
                self._hold_action = hold_action(action[-1])
            else:
                self._lifts += 1
                self._lift_left = self.lift_steps


def hold_action(gripper: float) -> np.ndarray:
    """Giữ tư thế và giữ nguyên gripper. Không dùng DUMMY_ACTION vì action đó mở gripper, sẽ làm rơi vật đang cầm."""
    action = np.zeros(len(ACTION_DIM_NAMES))
    action[-1] = gripper
    return action


def load_episodes(root: Path, tasks: list[int] | None = None) -> list[dict]:
    episodes = []
    for npz in sorted(root.glob("task_*/ep_*.npz")):
        with np.load(npz) as d:
            task_id = int(d["task_id"])
            if tasks and task_id not in tasks:
                continue
            monitor = FailureMonitor()
            prev = None
            rows = []
            for t in range(len(d["pred_env"])):
                rows.append(monitor.update(d["bin_entropy"][t], d["token_prob"][t], d["eef_state"][t], prev))
                prev = d["pred_env"][t]
            episodes.append(
                {
                    "task_id": task_id,
                    "episode": int(d["episode"]),
                    "success": bool(d["success"]),
                    "suite": str(d["suite"]),
                    "steps": len(rows),
                    # bỏ bước cuối: báo động ở đó có hiệu lực từ bước sau, mà bước sau không tồn tại
                    "scores": {s: np.array([r[s] for r in rows[:-1]]) for s in SIGNALS},
                }
            )
    return episodes


def conformal_threshold(scores: list[float], alpha: float) -> float:
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    return math.inf if rank > n else float(np.sort(scores)[rank - 1])


def fit_thresholds(calibration: list[dict], signals: tuple, alpha: float) -> dict:
    peaks = {s: [float(e["scores"][s].max(initial=-math.inf)) for e in calibration if e["success"]] for s in signals}
    return {s: conformal_threshold(peaks[s], alpha / len(signals)) for s in signals}


def first_alarm(episode: dict, thresholds: dict) -> int | None:
    steps = []
    for signal, threshold in thresholds.items():
        hits = episode["scores"][signal] > threshold
        if hits.any():
            steps.append(int(hits.argmax()))
    return min(steps, default=None)


def task_thresholds(episodes: list[dict], signals: tuple, alpha: float) -> dict:
    """{task_id: ngưỡng từ các task khác}, cộng "all" = ngưỡng từ mọi task, dùng khi áp sang suite khác."""
    out = {t: fit_thresholds([e for e in episodes if e["task_id"] != t], signals, alpha) for t in {e["task_id"] for e in episodes}}
    out["all"] = fit_thresholds(episodes, signals, alpha)
    return out


def alarms_for(episodes: list[dict], thresholds: dict, frozen: bool = False) -> list[int | None]:
    return [first_alarm(e, thresholds["all" if frozen else e["task_id"]]) for e in episodes]


def summarize(episodes: list[dict], alarms: list, timer_alarms: list) -> dict:
    """Lead = số bước báo sớm hơn timer; thất bại không được báo trước timer tính là 0."""
    fails = [(e, a, ta) for e, a, ta in zip(episodes, alarms, timer_alarms) if not e["success"]]
    n_success = sum(e["success"] for e in episodes)
    false_alarms = sum(a is not None for e, a in zip(episodes, alarms) if e["success"])
    leads = []
    for e, alarm, timer in fails:
        timer = e["steps"] if timer is None else timer
        leads.append((e["task_id"], 0 if alarm is None else max(timer - alarm, 0)))
    before_timer = [lead for _, lead in leads if lead > 0]
    return {
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarms / max(n_success, 1),
        "failures_alarmed": sum(a is not None for _, a, _ in fails),
        "failures_before_timer": len(before_timer),
        "median_lead_when_earlier": float(np.median(before_timer)) if before_timer else 0.0,
        "mean_lead": float(np.mean([lead for _, lead in leads])) if leads else 0.0,
        "mean_lead_ci95": bootstrap_lead(leads),
        "n_failures": len(fails),
        "n_successes": n_success,
    }


def bootstrap_lead(leads: list[tuple[int, int]]) -> list[float]:
    """Bootstrap theo task, không theo episode, vì thất bại trong cùng task tương quan với nhau."""
    by_task: dict[int, list[int]] = {}
    for task_id, lead in leads:
        by_task.setdefault(task_id, []).append(lead)
    groups = list(by_task.values())
    if not groups:
        return [0.0, 0.0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        picks = rng.integers(len(groups), size=len(groups))
        means.append(np.mean(np.concatenate([groups[i] for i in picks])))
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive, negative = positive[:, None], negative[None, :]
    return float((positive > negative).mean() + 0.5 * (positive == negative).mean())


def time_matched_auc(episodes: list[dict], signal: str) -> dict:
    """So tại cùng thời điểm, trên các episode còn chạy, để độ dài episode không tự tạo ra tín hiệu."""
    out = {}
    for step in AUC_STEPS:
        running = [e for e in episodes if e["steps"] - 1 > step]
        fail = [e["scores"][signal][: step + 1].max() for e in running if not e["success"]]
        ok = [e["scores"][signal][: step + 1].max() for e in running if e["success"]]
        out[step] = auc(np.array(fail), np.array(ok)) if fail and ok else None
    return out


def print_row(name: str, s: dict) -> None:
    low, high = s["mean_lead_ci95"]
    print(
        f"  {name:24} báo sai {s['false_alarms']:3d}/{s['n_successes']} ({s['false_alarm_rate'] * 100:4.1f}%)  "
        f"thất bại được báo {s['failures_alarmed']:3d}/{s['n_failures']}  trước timer {s['failures_before_timer']:3d}  "
        f"sớm hơn (trung vị) {s['median_lead_when_earlier']:5.1f}  lead TB {s['mean_lead']:5.1f} [{low:.1f}, {high:.1f}]"
    )


def to_json_thresholds(thresholds: dict) -> dict:
    return {str(k): {s: (None if math.isinf(v) else v) for s, v in row.items()} for k, row in thresholds.items()}


def from_json_thresholds(row: dict) -> dict:
    return {s: (math.inf if v is None else v) for s, v in row.items()}


def run_frozen(episodes: list[dict], config: dict) -> None:
    """Áp nguyên quy tắc đã chốt, không hiệu chỉnh lại trên dữ liệu này."""
    frozen = {"all": from_json_thresholds(config["thresholds"]["all"])}
    timer = {"all": from_json_thresholds(config["reference_timer"]["all"])}
    timer_alarms = alarms_for(episodes, timer, frozen=True)
    print(f"=== Quy tắc đã chốt: {'+'.join(frozen['all'])}, alpha={config['alpha']} ===")
    print_row("quy tắc đã chốt", summarize(episodes, alarms_for(episodes, frozen, frozen=True), timer_alarms))
    print_row("timer đã chốt", summarize(episodes, timer_alarms, timer_alarms))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True, help="thư mục chứa task_XX/ep_YYY.npz")
    ap.add_argument("--tasks", type=int, nargs="*", help="chỉ dùng các task id này (mặc định: tất cả)")
    ap.add_argument("--alarm-signals", nargs="+", choices=SIGNALS, default=["entropy", "timer"], help="tín hiệu dùng cho phần phản ứng")
    ap.add_argument("--alpha", type=float, default=0.05, help="tỉ lệ báo động sai mục tiêu cho phần phản ứng")
    ap.add_argument("--frozen", help="monitor.json của suite khác: chỉ kiểm tra quy tắc đó, không hiệu chỉnh lại")
    ap.add_argument("--out", help="mặc định: <rollouts>/monitor.json")
    args = ap.parse_args()

    root = Path(args.rollouts)
    episodes = load_episodes(root, args.tasks)
    if not episodes:
        ap.error(f"không có episode nào trong {root}")
    print(f"{len(episodes)} episode: {sum(e['success'] for e in episodes)} thành công, {sum(not e['success'] for e in episodes)} thất bại")
    if args.frozen:
        run_frozen(episodes, json.loads(Path(args.frozen).read_text()))
        return

    report = {}
    configs = [(s, (s,)) for s in SIGNALS] + [("+".join(UNION_SIGNALS), UNION_SIGNALS)]
    for alpha in ALPHAS:
        timer_alarms = alarms_for(episodes, task_thresholds(episodes, ("timer",), alpha))
        print(f"\n=== alpha = {alpha:.2f} ===")
        report[str(alpha)] = {}
        for name, signals in configs:
            alarms = alarms_for(episodes, task_thresholds(episodes, signals, alpha))
            report[str(alpha)][name] = summarize(episodes, alarms, timer_alarms)
            print_row(name, report[str(alpha)][name])

    print("\n=== AUC theo cùng thời điểm (thất bại so với thành công còn đang chạy) ===")
    aucs = {s: time_matched_auc(episodes, s) for s in SIGNALS if s != "timer"}
    for signal, row in aucs.items():
        print(f"  {signal:12} " + "  ".join(f"t={t}: {v:.3f}" for t, v in row.items() if v is not None))

    signals = tuple(args.alarm_signals)
    thresholds = task_thresholds(episodes, signals, args.alpha)
    timer = task_thresholds(episodes, ("timer",), args.alpha)
    alarms: dict[str, dict[str, int]] = {}
    for episode, alarm in zip(episodes, alarms_for(episodes, thresholds)):
        if alarm is not None:
            alarms.setdefault(str(episode["task_id"]), {})[str(episode["episode"])] = alarm
    config = {
        "suite": episodes[0]["suite"],
        "alpha": args.alpha,
        "signals": signals,
        "thresholds": to_json_thresholds(thresholds),
        "reference_timer": to_json_thresholds(timer),
        "alarms": alarms,
        "report": report,
        "time_matched_auc": aucs,
    }
    out = Path(args.out) if args.out else root / "monitor.json"
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"\n{sum(len(v) for v in alarms.values())} episode có báo động với {'+'.join(signals)}, alpha={args.alpha}")
    print(f"Đã lưu vào {out}")


if __name__ == "__main__":
    main()
