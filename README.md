# TDW Backend

Fork of [CoELA](https://github.com/UMass-Embodied-AGI/CoELA) adapted for the VLA multi-agent stack.

## What changed from upstream CoELA

- Stripped `cwah/` (VirtualHome), `assets/`, `doc/`, `detection_pipeline/`, `demo/`
- Migrated `gym` to `gymnasium`
- Added `VLABridgeAgent` -- replaces LLM-as-planner with physicai's VLM planner, keeps CoELA's spatial memory + A* pathfinding
- Added `vlm_adapter.py` -- bridges physicai VLM with CoELA prompt patterns
- Added `physicai_integration.py` -- entry points for DSM, fleet coordinator, contracts

## What we keep from CoELA

- `agent_memory.py` -- occupancy maps, depth-to-map projection, A* navigation
- `tdw_gym.py` -- gym-wrapped TDW environment with multi-agent support
- `transport_challenge_multi_agent/` -- scene loading, task infrastructure
- `dataset/` -- floorplan scenes and task definitions
- `LLM/` -- prompt templates (adapted for VLM)
- `scene_generator/` -- scene generation utilities

## Setup

```bash
cd tdw_mat
pip install -e .
```

Requires a running TDW build and the physicai root package on `PYTHONPATH`.

## Run

```bash
cd tdw_mat
bash scripts/test_VLA.sh
```

## Architecture

```
tdw_gym (simulation layer)
    |-- observations
VLABridgeAgent
    |-- AgentMemory (spatial maps, A* pathfinding)
    |-- physicai VLMPlanner (high-level decisions)
    |-- physicai DSM (gossip-based coordination)
    |-- physicai FleetCoordinator (task allocation)
    |-- discrete actions
tdw_gym.step()
```
