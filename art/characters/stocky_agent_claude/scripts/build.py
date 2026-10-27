"""Build the stocky agent. Options: see `charkit/buildlib.py`.

    blender -b --factory-startup -P scripts/build.py -- --turn --beauty all --save
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import agent  # noqa: E402
import buildlib  # noqa: E402

buildlib.run(os.path.normpath(os.path.join(HERE, "..")), "stocky_agent", agent.build, agent.make_materials, agent.BEAUTY,
             agent.HEIGHT, backdrop=agent.BACKDROP)
