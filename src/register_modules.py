"""
Register custom modules into Ultralytics' model parser namespace.

Call register_tomato_modules() before YOLO("custom.yaml").
"""

from __future__ import annotations

import importlib
from typing import Dict, Type

# 确保你的 tomato_modules.py 里确实有这些类的定义
from .tomato_modules import FeatureBalance, SimAM, ECA, TomatoCA, IdentityAttention


CUSTOM_MODULES: Dict[str, Type] = {
    "FeatureBalance": FeatureBalance,
    "SimAM": SimAM,
    "ECA": ECA,
    "TomatoCA": TomatoCA,
    "IdentityAttention": IdentityAttention,
}

# 👇👇👇 终极修复：函数原本的名字，必须和属性名一字不差！
def init_criterion_with_tomato_geometry(self, *args, **kwargs):
    pass


def register_tomato_modules(verbose: bool = False) -> None:
    """Expose custom modules to Ultralytics YAML parser."""
    task_module = importlib.import_module("ultralytics.nn.tasks")
    modules_module = importlib.import_module("ultralytics.nn.modules")

    # 1. 注册自定义的网络结构模块
    for name, cls in CUSTOM_MODULES.items():
        setattr(task_module, name, cls)
        setattr(modules_module, name, cls)
        if verbose:
            print(f"[register] {name} -> Ultralytics namespace")

    # 2. 修复带有几何损失（lambda_geo）的模型在保存和验证时的 Pickle 反序列化崩溃 Bug
    if hasattr(task_module, 'PoseModel'):
        # 将同名函数注入，完美符合 Pickle 的底层还原规则
        task_module.PoseModel.init_criterion_with_tomato_geometry = init_criterion_with_tomato_geometry