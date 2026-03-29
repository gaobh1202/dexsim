from .device import Device
from .keyboard import Keyboard

try:
    from .spacemouse import SpaceMouse
    from .dualsense import DualSense
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse or DualSense.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )

try:
    from .vr_device import VRDevice, VRTracker
except ImportError:
    # VR device is optional, so we don't raise an error if it fails to import
    pass

try:
    from .visionpro import VisionPro
except ImportError:
    # VisionPro device is optional, requires avp_stream module
    pass
