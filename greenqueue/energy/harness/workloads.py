"""Workload runners for the GPU sweep. Needs the ``gpu`` extra (torch, torchvision, transformers).

Each runner returns ``(step, flops_per_step)``. Data is random (energy depends on shapes, not
values), so nothing is downloaded except LLM weights, which the Colab notebook fetches.
Training workloads run one optimiser step per call; inference runs one ``generate`` per call.
FLOPs are counted once with ``torch.utils.flop_counter`` (forward + backward when training).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from greenqueue.energy.harness.sweep import RunConfig

Step = Callable[[], None]


def _dtype(torch: Any, precision: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def _count_flops(torch: Any, fn: Step) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc:
        fn()
    return float(fc.get_total_flops())


def make_runner(cfg: RunConfig, device: str = "cuda") -> tuple[Step, float]:
    import torch

    torch.manual_seed(0)
    extra = dict(cfg.extra)
    if cfg.workload == "resnet18":
        import torchvision

        model = torchvision.models.resnet18(num_classes=10).to(device)
        x = torch.randn(cfg.batch, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (cfg.batch,), device=device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.amp.GradScaler(enabled=cfg.precision == "fp16")

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=_dtype(torch, cfg.precision), enabled=cfg.precision != "fp32"
            ):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()

    elif cfg.workload == "vit_b16":
        import torchvision

        model = torchvision.models.vit_b_16(num_classes=1000).to(device)
        x = torch.randn(cfg.batch, 3, 224, 224, device=device)
        y = torch.randint(0, 1000, (cfg.batch,), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=_dtype(torch, cfg.precision)):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()

    elif cfg.workload == "bert_base":
        from transformers import BertConfig, BertForSequenceClassification

        model = BertForSequenceClassification(BertConfig(num_labels=2)).to(device)
        ids = torch.randint(0, 30000, (cfg.batch, cfg.seq), device=device)
        y = torch.randint(0, 2, (cfg.batch,), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=_dtype(torch, cfg.precision)):
                loss = model(input_ids=ids, labels=y).loss
            loss.backward()
            opt.step()
            torch.cuda.synchronize()

    elif cfg.workload == "llm_infer":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=_dtype(torch, cfg.precision)
        ).to(device)
        model.eval()
        prompt, gen = int(extra["prompt"]), int(extra["gen"])
        ids = torch.randint(100, 30000, (cfg.batch, prompt), device=device)

        def step() -> None:
            with torch.no_grad():
                model.generate(ids, max_new_tokens=gen, min_new_tokens=gen, do_sample=False)
            torch.cuda.synchronize()

    else:
        raise ValueError(f"unknown workload {cfg.workload}")

    flops = _count_flops(torch, step)
    return step, flops
