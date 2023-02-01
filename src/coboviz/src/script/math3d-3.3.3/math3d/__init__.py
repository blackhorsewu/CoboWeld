"""
Module for top level imports in PyMath3D.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2012-2017"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Production"

from .utils import set_precision
from .quaternion import Quaternion, Versor, UnitQuaternion
from .orientation import Orientation
from .orientation_computer import OrientationComputer
from .vector import Vector
from .transform import Transform
