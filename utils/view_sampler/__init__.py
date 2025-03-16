from typing import Any

from ..step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_bounded_v2 import ViewSamplerBoundedV2, ViewSamplerBoundedV2Cfg


VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
    "boundedv2": ViewSamplerBoundedV2,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
    | ViewSamplerBoundedV2Cfg
)
VIEW_SAMPLER_CFG_MAP = {
    "bounded": ViewSamplerBoundedCfg,
    "arbitrary": ViewSamplerArbitraryCfg,
    "evaluation": ViewSamplerEvaluationCfg,
    "all": ViewSamplerAllCfg,
    "boundedv2": ViewSamplerBoundedV2Cfg,
}

def get_view_sampler(
    cfg: ViewSamplerCfg | dict,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    if isinstance(cfg, dict):
        cfg = VIEW_SAMPLER_CFG_MAP[cfg['name']](**cfg)
    elif not isinstance(cfg, ViewSamplerCfg):
        raise TypeError(f"Expected cfg to be ViewSamplerCfg or dict, but got {type(cfg)}")
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
