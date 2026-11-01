# OpenVLA-7B trên LIBERO: infer, rollout, phân tích

Repo này dùng để chạy các checkpoint OpenVLA-7B đã được công bố, sinh trajectory action và phân tích đầu ra của model trên các suite LIBERO. Code được viết cho **server Linux có RTX 5090** (Blackwell, sm_120).

## Checkpoint nào dùng cho việc gì

| Checkpoint | Train trên | Dùng cho |
|---|---|---|
| `openvla/openvla-7b` | Open-X Embodiment | Thử infer với ảnh bất kỳ (`--unnorm-key bridge_orig`, ...). **Không có norm stats của LIBERO**, gần như không làm được task LIBERO |
| `openvla/openvla-7b-finetuned-libero-spatial` | LIBERO-Spatial | suite `libero_spatial` (paper: 84.7%) |
| `openvla/openvla-7b-finetuned-libero-object` | LIBERO-Object | suite `libero_object` (paper: 88.4%) |
| `openvla/openvla-7b-finetuned-libero-goal` | LIBERO-Goal | suite `libero_goal` (paper: 79.2%) |
| `openvla/openvla-7b-finetuned-libero-10` | LIBERO-10 (Long) | suite `libero_10` (paper: 53.7%) |

Các script tự chọn checkpoint fine-tune theo `--suite`. Muốn dùng checkpoint khác thì truyền `--checkpoint`.

## 1. Cài đặt trên server

Yêu cầu: driver NVIDIA ≥ 570 (hỗ trợ CUDA 12.8), conda, git, và thư viện EGL để MuJoCo render headless:

```bash
sudo apt install -y libegl1 libgl1 libglib2.0-0 ffmpeg   # nếu chưa có
git clone <repo-này> openvla-libero && cd openvla-libero
bash scripts/install_server.sh        # tạo env conda "openvla", cài torch cu128 + LIBERO, rồi chạy check_env.py
conda activate openvla
```

`scripts/check_env.py` phải báo `[ OK ]` cả 4 mục: GPU/torch, thư viện model, TensorFlow (CPU), LIBERO + MuJoCo EGL.

**Vì sao không theo hướng dẫn gốc của OpenVLA:** OpenVLA pin `torch==2.2.0` và `flash-attn==2.5.5`, cả hai đều không có kernel cho sm_120. Repo này dùng torch 2.7 cu128 và attention SDPA của PyTorch. Model được load thẳng từ HuggingFace (`trust_remote_code`), nên không cần cài package `openvla`. Phần tiền xử lý ảnh và hậu xử lý gripper được sao chép đúng từ `experiments/robot/` của OpenVLA, để kết quả so sánh được với paper.

## 2. Tải checkpoint và dataset

```bash
python -m ovla_eval.download --suites libero_spatial          # checkpoint (~15 GB) + dataset RLDS của suite
python -m ovla_eval.download --suites all                     # cả 4 suite
python -m ovla_eval.download --suites all --checkpoints --base # chỉ checkpoint, thêm openvla-7b gốc
```

Checkpoint được lưu vào HF cache (`~/.cache/huggingface`, đổi bằng `HF_HOME`). Dataset `openvla/modified_libero_rlds` được lưu vào `data/modified_libero_rlds/`. Đây là bản LIBERO OpenVLA đã regenerate để fine-tune: ảnh 256×256, đã bỏ các action no-op.

## 3. Infer một bước

```bash
# 1 frame trong dataset: in action dự đoán so với GT, kiểm tra decode khớp vla.predict_action()
python -m ovla_eval.infer_single --rlds-suite libero_spatial --episode 0 --step 30 --verify --save-input input.png

# ảnh bất kỳ với model gốc
python -m ovla_eval.infer_single --checkpoint openvla/openvla-7b --unnorm-key bridge_orig \
    --image frame.png --instruction "put the carrot on the plate"
```

Ở lần chạy đầu, hãy dùng `--verify` để chắc phần decode khớp với `predict_action()` của OpenVLA. Mở `input.png` để kiểm tra ảnh đầu vào đúng chiều, không bị lộn ngược.

## 4. Rollout trong simulator (thước đo chính)

```bash
python -m ovla_eval.rollout_libero --suite libero_spatial --trials 5      # chạy thử nhanh: 10 task × 5 episode
python -m ovla_eval.rollout_libero --suite libero_spatial                 # đầy đủ như paper: 10 task × 50 episode
python -m ovla_eval.rollout_libero --suite libero_goal --tasks 0 3 7 --trials 20
```

- Logic giống `run_libero_eval.py` của OpenVLA: center-crop 90%, chờ 10 bước đầu, cùng max steps, cùng init states, seed 7.
- Nếu chạy lại cùng `--out-dir`, các episode đã có `.npz` sẽ được bỏ qua (resume).
- Chạy lâu thì nên dùng `tmux`/`nohup`. Bản đầy đủ của một suite là 500 episode.

## 5. So sánh offline với ground truth

```bash
python -m ovla_eval.offline_eval --suite libero_spatial --max-episodes 20
python -m ovla_eval.offline_eval --suite libero_10 --max-episodes 10 --stride 2 --save-frames
```

Mỗi bước, model nhìn đúng ảnh trong demo (teacher forcing), rồi action dự đoán được so với action của người. Script in bảng MAE/RMSE theo từng chiều, MAE trong không gian chuẩn hoá (1 bin ≈ 0.0078), MAE của baseline đoán toàn 0, độ chính xác gripper, xác suất token và entropy trung bình.

> MAE offline thấp **không** đảm bảo rollout thành công, vì sai số tích luỹ trong closed-loop. Hãy dùng phần này để hiểu model sai ở chiều nào và lúc nào, còn success rate thì lấy từ phần 4.

## 6. Vẽ biểu đồ

```bash
python -m ovla_eval.plot outputs/offline/libero_spatial                 # + summary_mae.png
python -m ovla_eval.plot outputs/rollouts/libero_spatial/task_03 --limit 10
```

Mỗi episode có một PNG gồm 7 chiều action theo thời gian (dự đoán so với GT nếu có), cộng heatmap `p(token)` cho biết model tự tin đến đâu ở từng bước. Ảnh được lưu trong `<thư mục>/plots/`.

## 7. Giám sát thất bại và phản ứng sau báo động

Hai câu hỏi: tín hiệu có sẵn lúc chạy có báo thất bại sớm hơn timeout không khi kiểm soát tỉ lệ báo động sai, và sau báo động robot nên phản ứng thế nào. Bước 1 chỉ cần `.npz` nên chạy được không cần GPU. Bước 2 và 3 cần simulator. Riêng `mask` và `clip` cần GPU, mỗi phản ứng khoảng 3 giờ cho 500 episode.

```bash
# 1. Đặt ngưỡng trên 9 task, kiểm tra trên task còn lại, rồi so với timer
python -m ovla_eval.monitor --rollouts outputs/rollouts/libero_spatial --alarm-signals entropy timer
python -m ovla_eval.monitor --rollouts outputs/rollouts/libero_spatial --alarm-signals timer \
    --out outputs/rollouts/libero_spatial/monitor_timer.json

# 2. Baseline với tiếp xúc theo từng bước, và hai phản ứng dừng
python -m ovla_eval.replay_safety --suite libero_spatial --rollouts outputs/rollouts/libero_spatial \
    --out outputs/rollouts/libero_spatial/safety_steps.json
python -m ovla_eval.replay_safety --suite libero_spatial --rollouts outputs/rollouts/libero_spatial \
    --hold-after outputs/rollouts/libero_spatial/monitor.json --out outputs/responses/hold.json
python -m ovla_eval.replay_safety --suite libero_spatial --rollouts outputs/rollouts/libero_spatial \
    --hold-after outputs/rollouts/libero_spatial/monitor_timer.json --out outputs/responses/hold_timer.json

# 3. Hai phản ứng chạy tiếp sau khi nhấc tay
python -m ovla_eval.rollout_libero --suite libero_spatial --response mask --no-video \
    --thresholds outputs/rollouts/libero_spatial/monitor.json --out-dir outputs/responses/mask
python -m ovla_eval.rollout_libero --suite libero_spatial --response clip --no-video \
    --thresholds outputs/rollouts/libero_spatial/monitor.json --out-dir outputs/responses/clip
python -m ovla_eval.replay_safety --suite libero_spatial --rollouts outputs/responses/mask/libero_spatial
python -m ovla_eval.replay_safety --suite libero_spatial --rollouts outputs/responses/clip/libero_spatial

# 4. So sánh theo cặp
python -m ovla_eval.compare_responses --baseline outputs/rollouts/libero_spatial/safety_steps.json \
    --monitor outputs/rollouts/libero_spatial/monitor.json \
    --runs hold=outputs/responses/hold.json hold_timer=outputs/responses/hold_timer.json \
           mask=outputs/responses/mask/libero_spatial/safety_replay.json \
           clip=outputs/responses/clip/libero_spatial/safety_replay.json \
    --baseline-rollouts outputs/rollouts/libero_spatial \
    --prefix-check mask=outputs/responses/mask/libero_spatial clip=outputs/responses/clip/libero_spatial
```

- Báo động ở bước t thì phản ứng có hiệu lực từ bước t + 1.
- `hold` giữ tư thế và gripper tới hết episode. `mask` và `clip` nhấc tay với `dz >= 0.3` trong 10 bước, chờ 10 bước, tối đa 2 lần, rồi dừng hẳn.
- Nếu rollout gốc từng resume bằng process mới, truyền cùng `--fresh-env-at` cho `replay_safety`.
- Env có lịch sử giữa các episode, nên mỗi phản ứng phải chạy đủ các episode theo đúng thứ tự. `--prefix-check` báo bao nhiêu episode có đoạn trước báo động khớp baseline. Nếu không khớp hết thì phải ghi rõ điều này trong báo cáo.

## Định dạng đầu ra

```
outputs/
  rollouts/<suite>/
    summary.json                  # success rate theo task + tổng, kèm số liệu paper
    task_XX/ep_YYY.npz            # trajectory 1 episode
    task_XX/ep_YYY_{success,fail}.mp4
    monitor.json                  # ngưỡng theo task, báo động từng episode, bảng so sánh với timer
  offline/<suite>/
    summary.json                  # metric tổng + theo episode
    ep_ZZZZ.npz
```

Các mảng trong `.npz` (T = số bước model được gọi):

| key | shape | ý nghĩa |
|---|---|---|
| `pred_env` | (T, 7) | action thực sự gửi vào env, kể cả sau khi bị phản ứng thay đổi: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, gripper −1 = mở, +1 = đóng |
| `pred_raw` | (T, 7) | đầu ra `predict_action()` gốc, gripper ∈ [0, 1] với 1 = mở |
| `token_ids` | (T, 7) | token model sinh ra (256 token cuối vocab Llama-2 = 256 bin) |
| `token_prob` | (T, 7) | xác suất của token được chọn |
| `bin_entropy` | (T, 7) | entropy trên 256 bin (max ln 256 ≈ 5.55), cao = phân vân |
| `expected_raw` | (T, 7) | kỳ vọng action theo phân phối bin (soft decode) |
| `eef_state` | (T, 8) | vị trí EEF (3) + axis-angle (3) + 2 ngón gripper (2) |
| `gt_action` | (T, 7) | *chỉ có ở offline*: action trong demo |
| `success` | () | *chỉ có ở rollout* |
| `alarm_steps` | (K,) | *chỉ có khi rollout với `--response`*: các bước báo động |
| `frames` | (T, 224, 224, 3) | *offline với `--save-frames`* |

Đọc trong Python: `d = np.load("ep_000.npz"); d["pred_env"]`.

## Cách OpenVLA sinh action (để đọc kết quả)

1. Prompt là `In: What action should the robot take to {instruction}?\nOut:`, kèm một ảnh 224×224. Ảnh của LIBERO được xoay 180°, resize lanczos, rồi center-crop 90%.
2. Model sinh **7 token** cho **một bước**. OpenVLA không dự đoán chunk; trajectory được tạo bằng cách gọi lại model ở mỗi bước (closed-loop).
3. Mỗi token ứng với một trong 256 bin đều nhau trên [−1, 1]. Giá trị được un-normalize bằng quantile q01/q99 của dataset (`unnorm_key`). Gripper không được normalize và nằm trong [0, 1].
4. Gripper được binarize và đảo dấu theo quy ước LIBERO trước khi gửi vào env.

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `no kernel image is available for execution on the device` | torch không phải bản cu128. Chạy lại bước cài torch |
| `unnorm_key ... không có trong checkpoint` | Checkpoint không khớp suite. Dùng checkpoint fine-tune đúng suite, hoặc xem danh sách key được in ra |
| Treo ở lần import LIBERO đầu tiên | Chưa có `~/.libero/config.yaml`. Chạy lại `install_server.sh` |
| `WeightsUnpickler error` khi load init states | Thiếu `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Biến này đã được set trong `libero_utils.py`; nếu tự viết script riêng thì phải export |
| Lỗi EGL / `Cannot initialize a EGL device display` | Cài `libegl1`. Trên server nhiều GPU có thể thêm `MUJOCO_EGL_DEVICE_ID=0` |
| Success rate thấp hơn paper nhiều | Kiểm tra center crop (không truyền `--no-center-crop`), version `transformers/timm/tokenizers`, và `unnorm_key`. Kết quả giữa SDPA và flash-attn có thể lệch nhẹ vài % |
