from __future__ import annotations

from typing import Any, Dict, Optional


DEFAULT_RUNTIME_RETRIEVAL_PROFILE = "runtime_dense"
DEFAULT_EVAL_RETRIEVAL_PROFILE = "baseline_dense"


RETRIEVAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "runtime_dense": {
        "retrieval_mode": "dense",
        "metadata_filtering": False,
        "query_routing": False,
        "apply_filtering": True,
    },
    "baseline_dense": {
        "retrieval_mode": "dense",
        "metadata_filtering": False,
        "query_routing": False,
        "apply_filtering": False,
    },
    "dense_metadata_v1": {
        "retrieval_mode": "dense",
        "metadata_filtering": True,
        "query_routing": False,
        "apply_filtering": False,
    },
    "dense_routing_v1": {
        "retrieval_mode": "dense",
        "metadata_filtering": True,
        "query_routing": True,
        "apply_filtering": False,
    },
    "hybrid_v1": {
        "retrieval_mode": "hybrid",
        "metadata_filtering": False,
        "query_routing": False,
        "apply_filtering": True,
    },
}


def resolve_retrieval_profile(
    profile_name: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    name = (profile_name or DEFAULT_RUNTIME_RETRIEVAL_PROFILE).strip()
    if name not in RETRIEVAL_PROFILES:
        raise ValueError(f"Unknown retrieval profile: {name}")

    config = dict(RETRIEVAL_PROFILES[name])
    config["retrieval_profile"] = name

    for key, value in (overrides or {}).items():
        if key == "retrieval_profile":
            continue
        if value is not None:
            config[key] = value

    return config
