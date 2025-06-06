from data import cube_handling as CH
from system_simulation import forest
from rendering import blender_control as BC
from src.setup.runtime_environment import RuntimeEnvironment


def forest_pipe_test(runtime: RuntimeEnvironment, rng):
    """This is a testing box for all system_simulation canopy simulation funcionality.

    You'll have to run this several times. See inline comments what to run and what to
    comment out at each point.
    """

    # Leaf "measurement set" name for the demo. Can be uncommented all times
    set_name = "demo_leaves"

    # Generating low resolution random leaves. Uncomment for the first run, comment out for later runs.
    # LI.generate_prospect_leaf_random(set_name=set_name, leaf_count=2) # generate 2 random leaf spectra
    # LI.generate_prospect_leaf(set_name=set_name, sample_id=3, w=0.001) # add one dry default leaf spectra
    # new_sampling = [450,500,550,600,650,700,750,800,1450,1930] # wavelengths to be rendered
    # LI.resample_leaf_targets(set_name=set_name, new_sampling=new_sampling) # resample leaf spectra
    # LI.solve_leaf_material_parameters(set_name=set_name, clear_old_results=True) # run slab simulation

    # Some ID's and names. Can be uncommented all times
    # Scene IDs
    forest_id_master = "demo_forest_master"
    forest_id = "demo_forest"

    # Use pre-calculated soil spectra and default sun and sky spectra. They are automatically
    # interpolated to match the leaf spectra bands. Can be uncommented all times
    soil_name = "median_humid_clay_reflectance"
    sun_name = "default_sun"
    sky_name = "default_sky"

    # Here we create a new system_simulation scene from the template. Should be uncommented for the first run.
    # This creates a new "master" system_simulation you can use to generate other similar forests later.

    # Pack leaf data for system_simulation scene initialization. This can be uncommented all times
    leaves = [
        (set_name, 0, "Leaf material 1"),
        (set_name, 1, "Leaf material 2"),
        (set_name, 3, "Leaf material 3"),
    ]

    # system_simulation.init(leaves=leaves, conf_type='m2m', rng=rng,
    #             custom_forest_id=forest_id_master, soil_name=soil_name,
    #             sun_file_name=sun_name, sky_file_name=sky_name)

    # # Setup master and render preview
    # BC.setup_forest(forest_id=forest_id_master, leaf_name_list=['Leaf material 1', 'Leaf material 2', 'Leaf material 3'])
    # BC.render_forest(forest_id=forest_id_master, render_mode='preview')

    # Stop here. For the first run, everything after this should be commented out
    # Check the master file and make any changes before generating new "slave" system_simulation with random settings.
    # When you are happy with the new settings, uncomment the following (and comment out the previous lines as
    # instructed for second run).

    forest.init(
        leaves=leaves,
        conf_type="m2s",
        rng=rng,
        custom_forest_id=forest_id,
        copy_forest_id=forest_id_master,
        soil_name=soil_name,
        sun_file_name=sun_name,
        sky_file_name=sky_name,
    )

    # Running system_simulation.init only copies files. Running setup makes the Blender scene renderable.
    BC.setup_forest(
        forest_id=forest_id,
        leaf_name_list=["Leaf material 1", "Leaf material 2", "Leaf material 3"],
        runtime=runtime,
    )  # , 'Leaf material 4'])

    # Render bands for spectral cube along with additional images
    BC.render_forest(forest_id=forest_id, render_mode="preview", runtime=runtime)
    BC.render_forest(forest_id=forest_id, render_mode="visibility", runtime=runtime)
    BC.render_forest(forest_id=forest_id, render_mode="spectral", runtime=runtime)

    # Construct spectral cube in ENVI format
    CH.construct_envi_cube(system_sim_name=forest_id)
