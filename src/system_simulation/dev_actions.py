"""

This file contains actions that are meant only for developers!
Do not call these unless you know what you are doing.

"""

from src.rendering import blender_control as BC
from src.setup.runtime_environment import RuntimeEnvironment


def generate_global_master_forest_control_from_template(runtime: RuntimeEnvironment):
    """Writes a new global master control file.

    .. warning::
        This changes the control file that is in the repository as a base
        for all new simulation scenes. You should only call this if you
        make changes to the forest simulation template and need to have
        those changes reflected in the control file. Or if there is something
        that needs to fixed in the stored control file. Push the new control
        file to your repo after ensuring the new one works as expected.

    :param runtime: Runtime environment object that contains the Blender executable path.
    """

    BC.process_forest_control(runtime=runtime, global_master=True, generate=True)


def apply_global_master_forest_control_to_template(runtime: RuntimeEnvironment):
    """Apply a global master control file to the system simulation template.

    Counterpart of :py:func:`system_simulation.dev_actions.generate_global_master_forest_control_from_template`

    :param runtime: Runtime environment object that contains the Blender executable path.
    """

    BC.process_forest_control(runtime=runtime, global_master=True, generate=False)
