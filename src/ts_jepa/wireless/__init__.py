"""Wireless networking layer (separate from neural TS-JEPA)."""

from ts_jepa.wireless.channel import WirelessChannelModel
from ts_jepa.wireless.scheduler import ChannelAwareScheduler

__all__ = ["WirelessChannelModel", "ChannelAwareScheduler"]
