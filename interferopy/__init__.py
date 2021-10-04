from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

__all__ = ['cube', 'tools', 'casatools', 'casatools_vla_pipe']

from . import cube, tools, casatools, casatools_vla_pipe
