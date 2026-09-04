"""
DINOv3 -> YOLO26n 特征蒸馏训练脚本

适用情况：
1. 已经训练好了基础 YOLO26 / YOLO26n 模型 best.pt。
2. 已经下载好了 DINOv3 Hugging Face 格式文件：
   - config.json
   - model.safetensors
   - preprocessor_config.json
3. 这些文件和本脚本放在同一个文件夹。
4. 不修改 ultralytics / YOLO26 源代码。
5. DINOv3 只在训练时作为 teacher。
6. 验证和保存时自动切回 YOLO 原始 loss。
7. 训练完成后的 best.pt 仍然是普通 YOLO26n，可继续导出 ONNX / BPU bin。

运行：
    python train_dinov3_yolo26_distill.py
"""

import argparse
import math
import os
import shutil
import sys

os.environ["ULTRALYTICS_DISABLE_AUTOUPDATE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="DINOv3 -> YOLO26n feature distillation trainer"
    )
    parser.add_argument("--data", default=r"C:\Users\hlj23\Desktop\cable\dinov3_yolo26_complete\dinov3_yolo26_complete\final.ndjson", help="YOLO data.yaml / data file")
    parser.add_argument("--student", "--weights", dest="student", default=os.path.join(SCRIPT_DIR, "cable.pt"), help="Input YOLO26/YOLO26n .pt")
    parser.add_argument("--teacher", default=SCRIPT_DIR, help="DINOv3 Hugging Face model directory")
    parser.add_argument("--output", default="", help="Output distilled best.pt path")
    parser.add_argument("--project", default="runs_dinov3_distill", help="Ultralytics project directory")
    parser.add_argument("--name", default="yolo26n_dinov3_from", help="Ultralytics run name")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="", help="Training device, default auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kd-weight", type=float, default=0.01)
    parser.add_argument("--teacher-imgsz", type=int, default=320)
    parser.add_argument("--kd-size", type=int, default=20)
    parser.add_argument("--kd-start-epoch", type=int, default=5)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=15)
    return parser


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    build_arg_parser().parse_args()
    raise SystemExit(0)

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


# ============================================================
# 路径配置
# ============================================================

# 你的数据集 yaml
DATA_YAML = r"C:\Users\hlj23\Desktop\cable\dinov3_yolo26_complete\dinov3_yolo26_complete\final.ndjson"

# 你的基础 YOLO26 模型。
# 如果 best.pt 和本脚本在同一个文件夹，保持这样即可。
YOLO_WEIGHTS = os.path.join(SCRIPT_DIR, "cable.pt")

# 如果 best.pt 不在同一个文件夹，就改成这种绝对路径：
# YOLO_WEIGHTS = r"C:\Users\hlj23\Desktop\cable\new data --v1\cable test 2 no normal.v2i.yolo26\runs\detect\train\weights\best.pt"

# DINOv3 Hugging Face 文件目录。
# 当前目录下必须有：
# config.json / model.safetensors / preprocessor_config.json
DINO_TEACHER = SCRIPT_DIR

PROJECT = "runs_dinov3_distill"
NAME = "yolo26n_dinov3_from"


# ============================================================
# 训练参数：基于已经训练好的 600 张模型继续蒸馏，参数要保守
# ============================================================

EPOCHS = 30
IMGSZ = 640
BATCH = 4
WORKERS = 0
AMP = True

# 蒸馏参数
# 已经有 baseline 模型，所以蒸馏不要太猛
KD_WEIGHT = 0.01
TEACHER_IMGSZ = 320
KD_SIZE = 20
KD_START_EPOCH = 5

# 工业缺陷检测建议先关闭强增强
MOSAIC = 0.0
MIXUP = 0.0
COPY_PASTE = 0.0
CLOSE_MOSAIC = 0

# 微调学习率要小
LR0 = 0.0001
PATIENCE = 15


# ============================================================
# 检查函数
# ============================================================

def check_path_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 不存在：{path}")


def check_dino_hf_dir(path: str):
    required_files = [
        "config.json",
        "model.safetensors",
    ]

    missing = []
    for file_name in required_files:
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            missing.append(file_name)

    if missing:
        raise FileNotFoundError(
            f"DINO_TEACHER 目录缺少文件：{missing}\n"
            f"当前 DINO_TEACHER 目录：{path}\n\n"
            f"请确认这些文件和训练脚本在同一个文件夹：\n"
            f"  config.json\n"
            f"  model.safetensors\n"
            f"  preprocessor_config.json\n"
        )


# ============================================================
# YOLO 特征抓取
# ============================================================

class FeatureTap:
    """
    抓取 YOLO 中间 4D 特征图。
    不修改 YOLO 源码，只用 forward hook 临时监听。
    """

    def __init__(self, model: nn.Module):
        self.features = []
        self.handles = []

        layers = getattr(model, "model", None)

        if layers is None:
            raise RuntimeError("找不到 model.model，请确认 YOLO26 权重能被 ultralytics 正常加载。")

        # 每次 forward 开始前清空特征
        try:
            first_layer = layers[0]
            self.handles.append(first_layer.register_forward_pre_hook(self._pre_hook))
        except Exception:
            LOGGER.warning("无法给 YOLO 第一层挂 pre_hook，特征缓存可能无法自动清空。")

        # 给每一层挂 hook，收集中间 4D feature map
        for layer in layers:
            self.handles.append(layer.register_forward_hook(self._hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _pre_hook(self, module, inputs):
        self.features.clear()

    def _hook(self, module, inputs, output):
        def collect(x):
            if isinstance(x, torch.Tensor):
                if (
                    x.ndim == 4
                    and x.shape[1] >= 8
                    and x.shape[2] >= 4
                    and x.shape[3] >= 4
                ):
                    self.features.append(x)

            elif isinstance(x, (list, tuple)):
                for v in x:
                    collect(v)

            elif isinstance(x, dict):
                for v in x.values():
                    collect(v)

        collect(output)


# ============================================================
# DINOv3 特征提取
# ============================================================

def imagenet_normalize(img01: torch.Tensor) -> torch.Tensor:
    """
    DINO / ViT 常用 ImageNet mean/std。

    img01:
        RGB
        range 0~1
        shape [B, 3, H, W]
    """
    mean = img01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = img01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (img01 - mean) / std


def get_patch_size_from_config(dino: nn.Module) -> int:
    cfg = getattr(dino, "config", None)

    if cfg is None:
        return 16

    patch_size = getattr(cfg, "patch_size", None)

    if patch_size is None:
        return 16

    if isinstance(patch_size, (list, tuple)):
        return int(patch_size[0])

    return int(patch_size)


def get_num_register_tokens(dino: nn.Module) -> int:
    cfg = getattr(dino, "config", None)

    if cfg is None:
        return 0

    return int(getattr(cfg, "num_register_tokens", 0))


def dino_patch_features(
    dino: nn.Module,
    img01: torch.Tensor,
    teacher_imgsz: int,
    kd_size: int,
) -> torch.Tensor:
    """
    提取 DINOv3 patch features。

    输入：
        img01: [B, 3, H, W]，RGB，0~1

    输出：
        teacher_feat: [B, C, N]
        N = kd_size * kd_size
    """

    img01 = F.interpolate(
        img01,
        size=(teacher_imgsz, teacher_imgsz),
        mode="bilinear",
        align_corners=False,
    )

    pixel_values = imagenet_normalize(img01).float()

    out = dino(pixel_values=pixel_values)

    if not hasattr(out, "last_hidden_state"):
        raise RuntimeError("DINO 输出中没有 last_hidden_state，请检查 transformers 版本或模型格式。")

    hs = out.last_hidden_state

    patch_size = get_patch_size_from_config(dino)
    num_register = get_num_register_tokens(dino)

    # DINOv3 ViT 通常是：[CLS] + register tokens + patch tokens
    patch_tokens = hs[:, 1 + num_register :, :]

    b, n, c = patch_tokens.shape

    gh = pixel_values.shape[-2] // patch_size
    gw = pixel_values.shape[-1] // patch_size
    expected_n = gh * gw

    # 如果 config 里的 register token 数不匹配，则自动寻找 patch token 起点
    if n != expected_n:
        found = False

        for skip in range(1, 10):
            candidate = hs[:, skip:, :]
            cn = candidate.shape[1]
            side = int(math.sqrt(cn))

            if side * side == cn:
                patch_tokens = candidate
                b, n, c = patch_tokens.shape
                gh = side
                gw = side
                found = True
                break

        if not found:
            raise RuntimeError(
                f"DINO patch 数不匹配。\n"
                f"got n={n}, expected={expected_n}\n"
                f"last_hidden_state shape={tuple(hs.shape)}\n"
                f"patch_size={patch_size}, num_register={num_register}"
            )

    feat = patch_tokens.transpose(1, 2).reshape(b, c, gh, gw).float()

    feat = F.interpolate(
        feat,
        size=(kd_size, kd_size),
        mode="bilinear",
        align_corners=False,
    )

    return feat.flatten(2)


# ============================================================
# 蒸馏 loss
# ============================================================

def pick_student_feature(features, kd_size: int) -> torch.Tensor:
    """
    从 YOLO 中间特征里挑一个空间尺寸最接近 kd_size 的特征图。

    对 640 输入来说：
    P3 通常是 80x80
    P4 通常是 40x40
    P5 通常是 20x20

    DINO 320 输入、patch size 16 时是 20x20。
    所以通常会选到 YOLO 的 P5 特征。
    """

    valid = [
        t for t in features
        if isinstance(t, torch.Tensor)
        and t.ndim == 4
        and t.shape[2] >= 4
        and t.shape[3] >= 4
    ]

    if not valid:
        raise RuntimeError("没有抓到 YOLO 中间特征；可能是 hook 没有挂上。")

    return min(
        valid,
        key=lambda t: abs(t.shape[-2] - kd_size) + abs(t.shape[-1] - kd_size)
    )


def gram_distill_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    kd_size: int,
) -> torch.Tensor:
    """
    Gram / 关系蒸馏。

    优点：
    不要求 YOLO 通道数等于 DINO 通道数。
    只要求空间 patch 之间的关系相似。
    """

    s = F.interpolate(
        student_feat.float(),
        size=(kd_size, kd_size),
        mode="bilinear",
        align_corners=False,
    )

    s = s.flatten(2)          # [B, Cs, N]
    t = teacher_feat.float()  # [B, Ct, N]

    s = F.normalize(s, dim=1, eps=1e-6)
    t = F.normalize(t, dim=1, eps=1e-6)

    gs = torch.bmm(s.transpose(1, 2), s) / math.sqrt(max(s.shape[1], 1))
    gt = torch.bmm(t.transpose(1, 2), t) / math.sqrt(max(t.shape[1], 1))

    return F.mse_loss(gs, gt)


class DistillLossWrapper:
    """
    包一层 YOLO 原始 loss，在它基础上加 DINOv3 KD loss。

    关键点：
    1. 训练模型中使用它。
    2. EMA / deepcopy 时不能复制 trainer / dino / dataloader，否则会报：
       _SingleProcessDataLoaderIter cannot be pickled
    3. 所以实现 __getstate__ / __setstate__，让 deepcopy 只复制一个空壳。
    """

    def __init__(self, trainer, old_loss, dino, tap):
        self.trainer = trainer
        self.old_loss = old_loss
        self.dino = dino
        self.tap = tap
        self._last_log_epoch = -1
        self._is_ema_copy = False

    def __getstate__(self):
        """
        Ultralytics 创建 EMA 时会 deepcopy(model)。
        这里禁止 deepcopy 时复制 trainer / dino / dataloader。
        """
        return {
            "_is_ema_copy": True,
            "_last_log_epoch": -1,
        }

    def __setstate__(self, state):
        """
        EMA / deepcopy 出来的 wrapper 不能再用于蒸馏训练。
        """
        self.trainer = None
        self.old_loss = None
        self.dino = None
        self.tap = None
        self._last_log_epoch = -1
        self._is_ema_copy = True

    def __call__(self, batch, preds=None):
        if getattr(self, "_is_ema_copy", False):
            raise RuntimeError(
                "EMA copy 中的 DistillLossWrapper 不应该被调用。"
                "验证阶段应该先移除蒸馏 loss，只使用 YOLO 原始 loss。"
            )

        # YOLO 原始检测 loss
        det_loss, loss_items = self.old_loss(batch, preds)

        epoch = int(getattr(self.trainer, "epoch", 0))

        if KD_WEIGHT <= 0 or epoch < KD_START_EPOCH:
            return det_loss, loss_items

        imgs = batch["img"]

        # YOLO student feature
        student_feat = pick_student_feature(self.tap.features, KD_SIZE)

        # DINO teacher feature
        with torch.no_grad():
            if next(self.dino.parameters()).device != imgs.device:
                self.dino.to(imgs.device)

            teacher_feat = dino_patch_features(
                dino=self.dino,
                img01=imgs.detach(),
                teacher_imgsz=TEACHER_IMGSZ,
                kd_size=KD_SIZE,
            )

        kd_loss = gram_distill_loss(
            student_feat=student_feat,
            teacher_feat=teacher_feat,
            kd_size=KD_SIZE,
        )

        total_loss = det_loss + KD_WEIGHT * kd_loss

        if epoch % 5 == 0 and self._last_log_epoch != epoch:
            LOGGER.info(
                f"DINOv3 KD loss={float(kd_loss.detach().cpu()):.6f}, "
                f"weight={KD_WEIGHT}"
            )
            self._last_log_epoch = epoch

        return total_loss, loss_items


# ============================================================
# 自定义 Trainer
# ============================================================

class DINOv3DistillTrainer(DetectionTrainer):
    """
    外部自定义 Trainer。
    不改 ultralytics 源码。

    核心逻辑：
    1. 训练时使用 YOLO loss + DINOv3 KD loss。
    2. 验证时临时移除蒸馏 loss，只用 YOLO 原始 loss。
    3. 保存 checkpoint 时临时移除蒸馏 loss，避免 DINOv3 teacher 被保存进 best.pt。
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        from transformers import AutoModel

        check_dino_hf_dir(DINO_TEACHER)

        LOGGER.info(f"Loading local DINOv3 teacher from: {DINO_TEACHER}")

        dino = AutoModel.from_pretrained(
            DINO_TEACHER,
            local_files_only=True,
        )

        dino.eval()

        for p in dino.parameters():
            p.requires_grad_(False)

        tap = FeatureTap(model)
        old_loss = model.loss

        wrapper = DistillLossWrapper(
            trainer=self,
            old_loss=old_loss,
            dino=dino,
            tap=tap,
        )

        # 给训练模型挂蒸馏 loss
        model.loss = wrapper

        self._distill_loss_wrapper = wrapper
        self._distill_train_model = model
        self._distill_old_loss = old_loss

        LOGGER.info(
            f"DINOv3 KD enabled: weight={KD_WEIGHT}, teacher_imgsz={TEACHER_IMGSZ}, "
            f"kd_size={KD_SIZE}, start_epoch={KD_START_EPOCH}"
        )

        return model

    @staticmethod
    def _pop_instance_distill_loss(model):
        """
        如果 model.loss 是实例属性里的 DistillLossWrapper，就临时删除它。
        删除后，Python 会重新使用 YOLO 模型类本身的原始 loss 方法。
        """
        if model is None:
            return None

        try:
            current_loss = vars(model).get("loss", None)
        except Exception:
            current_loss = None

        if isinstance(current_loss, DistillLossWrapper):
            try:
                delattr(model, "loss")
            except Exception:
                pass
            return current_loss

        return None

    @staticmethod
    def _restore_instance_distill_loss(model, wrapper):
        """
        把训练模型的蒸馏 loss 恢复回去。
        """
        if model is not None and wrapper is not None:
            model.loss = wrapper

    def validate(self):
        """
        验证阶段不要用 DINOv3 KD loss。
        否则 EMA / deepcopy 后的模型可能带着蒸馏 wrapper，导致验证报错。
        """
        train_model = getattr(self, "model", None)
        train_wrapper = self._pop_instance_distill_loss(train_model)

        ema_model = None
        ema_wrapper = None

        ema_obj = getattr(self, "ema", None)
        if ema_obj is not None and getattr(ema_obj, "ema", None) is not None:
            ema_model = ema_obj.ema
            ema_wrapper = self._pop_instance_distill_loss(ema_model)

        try:
            return super().validate()
        finally:
            # 训练模型恢复蒸馏 loss，后续 epoch 继续训练要用
            self._restore_instance_distill_loss(train_model, train_wrapper)

            # EMA 模型不恢复蒸馏 loss
            _ = ema_model
            _ = ema_wrapper

    def save_model(self, *args, **kwargs):
        """
        保存 checkpoint 时，也临时移除蒸馏 loss。
        避免 best.pt / last.pt 里保存 DINOv3 teacher。
        """
        train_model = getattr(self, "model", None)
        train_wrapper = self._pop_instance_distill_loss(train_model)

        ema_model = None
        ema_wrapper = None

        ema_obj = getattr(self, "ema", None)
        if ema_obj is not None and getattr(ema_obj, "ema", None) is not None:
            ema_model = ema_obj.ema
            ema_wrapper = self._pop_instance_distill_loss(ema_model)

        try:
            return super().save_model(*args, **kwargs)
        finally:
            # 继续训练还需要恢复训练模型的蒸馏 loss
            self._restore_instance_distill_loss(train_model, train_wrapper)

            # EMA 模型不恢复蒸馏 loss
            _ = ema_model
            _ = ema_wrapper


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    DATA_YAML = args.data
    YOLO_WEIGHTS = args.student
    DINO_TEACHER = args.teacher
    PROJECT = args.project
    NAME = args.name
    EPOCHS = args.epochs
    IMGSZ = args.imgsz
    BATCH = args.batch
    WORKERS = args.workers
    AMP = args.amp
    KD_WEIGHT = args.kd_weight
    TEACHER_IMGSZ = args.teacher_imgsz
    KD_SIZE = args.kd_size
    KD_START_EPOCH = args.kd_start_epoch
    LR0 = args.lr0
    PATIENCE = args.patience

    print("=" * 70)
    print(f"脚本目录: {SCRIPT_DIR}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    device = args.device or (0 if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU 训练；DINOv3 蒸馏会非常慢。")

    print("=" * 70)

    check_path_exists(DATA_YAML, "DATA_YAML")
    check_path_exists(YOLO_WEIGHTS, "YOLO_WEIGHTS")
    check_path_exists(DINO_TEACHER, "DINO_TEACHER")
    check_dino_hf_dir(DINO_TEACHER)

    print("数据集：", DATA_YAML)
    print("YOLO 初始权重：", YOLO_WEIGHTS)
    print("DINOv3 teacher：", DINO_TEACHER)
    print("输出目录：", os.path.join(PROJECT, NAME))
    print("=" * 70)

    model = YOLO(YOLO_WEIGHTS)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=WORKERS,
        amp=AMP,
        pretrained=True,
        verbose=True,
        project=PROJECT,
        name=NAME,
        lr0=LR0,
        patience=PATIENCE,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        close_mosaic=CLOSE_MOSAIC,
        trainer=DINOv3DistillTrainer,
    )

    best_pt = os.path.join(PROJECT, NAME, "weights", "best.pt")
    if args.output:
        check_path_exists(best_pt, "DINOv3 distilled best.pt")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        shutil.copy2(best_pt, args.output)
        print("已复制蒸馏输出：", args.output)

    print("=" * 70)
    print("DINOv3 蒸馏训练完成。")
    print("best.pt 通常在：")
    print(best_pt)
    print("=" * 70)
