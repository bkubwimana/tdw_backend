"""
PhysicAI Integration Module

Top-level entry point for running CoELA environments with physicai's
VLA/coordination stack. Handles sys.path setup and provides factory
functions for creating integrated components.

Usage:
    from physicai_integration import create_dsm, create_vlm_adapter
"""

import os
import sys

PHYSICAI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PHYSICAI_ROOT not in sys.path:
    sys.path.insert(0, PHYSICAI_ROOT)


def create_vlm_adapter(url=None):
    """Create VLM adapter for plan generation."""
    from LLM.vlm_adapter import VLMAdapter
    return VLMAdapter(vlm_url=url)


def create_dsm(gossip_period_ms=50.0, max_aoi_ms=3000.0):
    """Create DSM instance for multi-agent coordination."""
    try:
        from simulator.coordination.dsm import DistributedSharedMemory
        return DistributedSharedMemory(
            gossip_period_ms=gossip_period_ms,
            max_aoi_ms=max_aoi_ms
        )
    except ImportError:
        return None


def create_fleet_coordinator(**kwargs):
    """Create fleet coordinator for task allocation."""
    try:
        from simulator.coordination.fleet import FleetCoordinator
        return FleetCoordinator(**kwargs)
    except ImportError:
        return None


def create_vla_interface(model_name="hierarchical", **kwargs):
    """Create a physicai VLA model instance."""
    try:
        from simulator.vla.interface import create_vla
        return create_vla(model_name, **kwargs)
    except ImportError:
        return None


def get_timing_contract(sensor_period_ms=33.0, loop_period_ms=100.0):
    """Get timing contract for the integrated system."""
    try:
        from contracts.formal.timing import TimingContract
        return TimingContract.control_loop_contract(
            sensing_latency_ms=sensor_period_ms,
            computation_latency_ms=50.0,
            actuation_latency_ms=20.0,
            loop_period_ms=loop_period_ms
        )
    except ImportError:
        return None
