"""Load the installed pySOT surrogate without unrelated DOE imports.

pySOT 0.3.3's package initializer imports pyDOE2, whose unused `imp` import fails
on Python >=3.12. A private package namespace keeps the original surrogate and
utilities unchanged, without monkeypatching sys.modules['pySOT'] or site-packages.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path

name = __package__ + '._pysot'
if name not in sys.modules:
    spec = importlib.util.find_spec('pySOT')
    if spec is None:
        raise ImportError('Install requirements-amq.txt before running AMQ')
    package = types.ModuleType(name)
    package.__path__ = [str(Path(spec.origin).parent)]
    package.__package__ = name
    sys.modules[name] = package
surrogate = importlib.import_module(name + '.surrogate')
RBFInterpolant = surrogate.RBFInterpolant
CubicKernel = surrogate.CubicKernel
TPSKernel = surrogate.TPSKernel
LinearTail = surrogate.LinearTail
ConstantTail = surrogate.ConstantTail
