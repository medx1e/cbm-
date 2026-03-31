"""CBM-V1 configuration."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class CBMConfig:
    """Configuration for the Concept Bottleneck Model V1.

    Controls bottleneck dimensions, concept loss weighting, and
    frozen-vs-joint training mode.
    """

    # Number of concept outputs (must match concept registry size)
    num_concepts: int = 11

    # Hidden layer sizes for the concept head (encoder_output → concepts)
    concept_head_hidden_sizes: tuple[int, ...] = (64,)

    # Hidden layer sizes for the actor FC after bottleneck
    actor_hidden_sizes: tuple[int, ...] = (64, 32)

    # Hidden layer sizes for the critic FC after bottleneck
    critic_hidden_sizes: tuple[int, ...] = (64, 32)

    # Weight of concept supervision loss in total loss
    lambda_concept: float = 0.1

    # Training mode: "frozen" = encoder frozen, heads train;
    #                "joint" = everything trains together
    mode: Literal["frozen", "joint"] = "frozen"

    # Number of twin Q-networks for the critic
    num_critics: int = 2

    # Whether critic networks share the encoder
    shared_encoder: bool = False

    # Which concept phases to use from the registry
    concept_phases: tuple[int, ...] = (1, 2)

    # Concept names (populated at runtime from registry)
    # This is informational — the actual concepts come from the registry
    concept_names: tuple[str, ...] = (
        "ego_speed",
        "ego_acceleration",
        "dist_nearest_object",
        "num_objects_within_10m",
        "traffic_light_red",
        "dist_to_traffic_light",
        "heading_deviation",
        "progress_along_route",
        "ttc_lead_vehicle",
        "lead_vehicle_decelerating",
        "at_intersection",
    )

    # Binary concept indices (for BCE loss)
    binary_concept_indices: tuple[int, ...] = (4, 9, 10)
    # Continuous concept indices (for Huber loss)
    continuous_concept_indices: tuple[int, ...] = (0, 1, 2, 3, 5, 6, 7, 8)
