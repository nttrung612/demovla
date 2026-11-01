"""Wrapper quanh checkpoint OpenVLA trên HuggingFace.

Ngoài action cuối cùng (giống vla.predict_action()), còn trả về token, xác suất và entropy của từng chiều
action để phân tích model "chắc chắn" đến đâu.

Cách OpenVLA mã hoá action: mỗi chiều được chuẩn hoá về [-1, 1] bằng quantile q01/q99 của dataset,
chia thành 256 bin, mỗi bin ứng với 1 trong 256 token cuối của vocab Llama-2. Model sinh 7 token liên tiếp.
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, LogitsProcessor, LogitsProcessorList

from .constants import ACTION_DIM_NAMES
from .preprocess import center_crop

PROMPT_TEMPLATE = "In: What action should the robot take to {}?\nOut:"
EMPTY_TOKEN_ID = 29871  # token '' mà OpenVLA chèn sau "Out:" lúc train
N_ACTION_TOKENS = 256
DZ_INDEX = ACTION_DIM_NAMES.index("dz")


@dataclass
class ActionPrediction:
    raw: np.ndarray  # (D,) action đã un-normalize, gripper ∈ [0, 1] với 1 = mở (= vla.predict_action())
    env: np.ndarray  # (D,) action gửi vào LIBERO: gripper ∈ {-1 (mở), +1 (đóng)}
    token_ids: np.ndarray  # (D,) token được sinh (greedy)
    bins: np.ndarray  # (D,) chỉ số bin [0, 254]
    token_prob: np.ndarray  # (D,) xác suất softmax của token được chọn
    bin_entropy: np.ndarray  # (D,) entropy (nats) của phân phối trên 256 token action, max = ln 256 ≈ 5.55
    expected_raw: np.ndarray  # (D,) kỳ vọng action theo phân phối bin (soft decode), để so với argmax


class TokenMaskProcessor(LogitsProcessor):
    """Chỉ cho phép các token trong `allowed` ở chiều action `dim`, các chiều khác không đổi.

    Giữ lại logits gốc của mọi chiều: out.scores là logits đã qua processor, nên nếu dùng nó thì entropy và xác suất
    sẽ mô tả phân phối đã che, không còn là phân phối thật của model.
    """

    def __init__(self, allowed: torch.Tensor, dim: int, prompt_len: int):
        self.allowed = allowed  # (vocab,) bool
        self.dim = dim
        self.prompt_len = prompt_len
        self.raw_scores: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.raw_scores.append(scores.clone())
        if input_ids.shape[1] - self.prompt_len == self.dim:
            return scores.masked_fill(~self.allowed, -torch.inf)
        return scores


def gripper_to_env(action: np.ndarray) -> np.ndarray:
    """normalize_gripper_action(binarize=True) + invert_gripper_action của OpenVLA:
    [0, 1] (1 = mở) -> {-1, +1} -> đảo dấu để ra quy ước LIBERO (-1 = mở, +1 = đóng)."""
    out = np.array(action, dtype=np.float64, copy=True)
    out[..., -1] = -np.sign(2.0 * out[..., -1] - 1.0)
    return out


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OpenVLAPolicy:
    def __init__(
        self,
        checkpoint: str,
        unnorm_key: str | None = None,
        device: str = "cuda:0",
        attn_implementation: str = "sdpa",
        use_center_crop: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.use_center_crop = use_center_crop
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.vla = (
            AutoModelForVision2Seq.from_pretrained(
                checkpoint,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.unnorm_key = self.resolve_unnorm_key(unnorm_key)
        stats = self.vla.get_action_stats(self.unnorm_key)
        self.q01 = np.asarray(stats["q01"], dtype=np.float64)
        self.q99 = np.asarray(stats["q99"], dtype=np.float64)
        self.mask = np.asarray(stats.get("mask", np.ones_like(self.q01, dtype=bool)), dtype=bool)
        self.action_dim = len(self.q01)

        # 256 token action là [vocab-256, vocab-1]; token t <-> bin clip(vocab - t - 1, 0, 254)
        self.vocab_size = int(self.vla.vocab_size)
        self.bin_centers = np.asarray(self.vla.bin_centers, dtype=np.float64)
        self._action_token_ids = torch.arange(self.vocab_size - N_ACTION_TOKENS, self.vocab_size, device=self.device)
        action_token_bins = self._token_to_bin(self._action_token_ids.cpu().numpy())
        self._action_token_centers = torch.tensor(self.bin_centers[action_token_bins], dtype=torch.float32, device=self.device)

    @property
    def norm_stat_keys(self) -> list[str]:
        return list(self.vla.norm_stats)

    def resolve_unnorm_key(self, key: str | None) -> str:
        keys = self.norm_stat_keys
        if key is None:
            if len(keys) == 1:
                return keys[0]
            raise ValueError(f"Checkpoint có nhiều bộ norm stats, cần chọn --unnorm-key trong: {keys}")
        for candidate in (key, f"{key}_no_noops"):
            if candidate in keys:
                return candidate
        raise ValueError(f"unnorm_key '{key}' không có trong checkpoint {self.checkpoint}. Có: {keys}")

    def _token_to_bin(self, token_ids: np.ndarray) -> np.ndarray:
        return np.clip(self.vocab_size - token_ids - 1, 0, self.bin_centers.shape[0] - 1)

    def unnormalize(self, normalized: np.ndarray) -> np.ndarray:
        """[-1, 1] -> đơn vị gốc của dataset. Chiều có mask=False (gripper) giữ nguyên."""
        return np.where(self.mask, 0.5 * (normalized + 1.0) * (self.q99 - self.q01) + self.q01, normalized)

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Ngược lại của unnormalize, dùng để đo sai số trong không gian model nhìn thấy."""
        return np.where(self.mask, 2.0 * (action - self.q01) / (self.q99 - self.q01) - 1.0, action)

    def _prepare_inputs(self, image: np.ndarray, instruction: str):
        """image: uint8 (224, 224, 3) đã resize như lúc train (xem preprocess.resize_image)."""
        if self.use_center_crop:
            image = center_crop(image)
        prompt = PROMPT_TEMPLATE.format(instruction.lower())
        return self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)

    def _dz_allowed_tokens(self, dz_min: float) -> torch.Tensor:
        """(vocab,) bool: các token action của dz có giá trị >= dz_min. Không token nào đạt thì lấy token dz lớn nhất."""
        centers = self._action_token_centers.cpu().numpy().astype(np.float64)
        q01, q99 = self.q01[DZ_INDEX], self.q99[DZ_INDEX]
        values = torch.as_tensor(0.5 * (centers + 1.0) * (q99 - q01) + q01, device=self.device)
        allowed = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        ok = values >= dz_min
        allowed[self._action_token_ids[ok if ok.any() else values == values.max()]] = True
        return allowed

    @torch.inference_mode()
    def predict(self, image: np.ndarray, instruction: str, dz_min: float | None = None) -> ActionPrediction:
        """dz_min: nếu có, chỉ decode token dz có giá trị >= dz_min. Các chiều sau dz được sinh tiếp theo token đã che.
        token_prob, bin_entropy và expected_raw luôn lấy từ phân phối gốc của model."""
        inputs = self._prepare_inputs(image, instruction)
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
            extra = torch.full_like(input_ids[:, :1], EMPTY_TOKEN_ID)
            input_ids = torch.cat([input_ids, extra], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(extra)], dim=1)

        mask = None
        extra = {}
        if dz_min is not None:
            mask = TokenMaskProcessor(self._dz_allowed_tokens(dz_min), DZ_INDEX, input_ids.shape[1])
            extra["logits_processor"] = LogitsProcessorList([mask])
        out = self.vla.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.action_dim,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            **extra,
        )
        token_ids = out.sequences[0, -self.action_dim :].cpu().numpy()
        logits = torch.stack(mask.raw_scores if mask else out.scores, dim=0)[:, 0].float()  # (D, vocab)

        rows = torch.arange(self.action_dim, device=self.device)
        token_prob = torch.softmax(logits, dim=-1)[rows, torch.as_tensor(token_ids, device=self.device)]
        p_bins = torch.softmax(logits[:, self._action_token_ids], dim=-1)  # (D, 256)
        entropy = -(p_bins * torch.log(p_bins.clamp_min(1e-12))).sum(dim=-1)
        expected_norm = (p_bins @ self._action_token_centers).cpu().numpy().astype(np.float64)

        bins = self._token_to_bin(token_ids)
        raw = self.unnormalize(self.bin_centers[bins])
        return ActionPrediction(
            raw=raw,
            env=gripper_to_env(raw),
            token_ids=token_ids,
            bins=bins,
            token_prob=token_prob.cpu().numpy().astype(np.float64),
            bin_entropy=entropy.cpu().numpy().astype(np.float64),
            expected_raw=self.unnormalize(expected_norm),
        )

    @torch.inference_mode()
    def predict_reference(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Gọi thẳng vla.predict_action() của OpenVLA, dùng để kiểm tra predict() decode đúng."""
        inputs = self._prepare_inputs(image, instruction)
        return np.asarray(self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False), dtype=np.float64)
