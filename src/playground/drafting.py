import slab_model.training_data
from setup import initialization
from src.reflectance_lab import diffuse_reflectance
from src.slab_model import interface as SI, training_data as TD
from src.utils import spectra_utils as SU
from src import plotter
from src.data import toml_handling as TH
from publication_run import run_paper_tests, asym_test
from src.system_simulation import forest


if __name__ == "__main__":

    runtime = initialization.initialize()

    SI.visualize_leaf_models(
        training_set_name="train_iter_1", show_plot=False, plot_surf=True, plot_nn=False
    )

    run_paper_tests()

    asym_test(smthng="const_r_var_t")
    asym_test(smthng="const_t_var_r")

    set_name_iter_4 = "train_iter_4"
    # Training data visualization
    TD.visualize_training_data_pruning(set_name=set_name_iter_4, show=True)
    SI.visualize_leaf_models(training_set_name=set_name_iter_4, show_plot=True)
    SI.visualize_leaf_models(
        training_set_name=set_name_iter_4, show_plot=True, plot_nn=True
    )
    plotter._plot_starting_guess_coeffs_fitting(dont_show=False)

    # Let redo starting guess
    slab_model.training_data.generate_starting_guess()
    slab_model.training_data.fit_starting_guess_coefficients(degree=12)
    plotter._plot_starting_guess_coeffs_fitting(dont_show=False)

    # gsv.visualize_default_soils(save=False, dont_show=False)
    # gsv._write_default_soils()

    # plotter.plot_resampling(set_name='low_res')

    forest_id = forest.init()

    forest_id = "1406231352"

    # Sun power test
    diffuse_reflectance.run(runtime=runtime, data_exits=True)

    leaf_stuff = [("try_random_p_leaves", 0, 1), ("try_random_p_leaves", 1, 3)]
    forest.init(copy_forest_id="0102231033")

    # Let's first generate some random PROSPECT leaves
    set_name = "try_random_p_leaves"
    # generates three leaf targets to \HyperBlend\leaf_measurement_sets\try_random_p_leaves\sample_targets
    SI.generate_prospect_leaf_random(set_name=set_name, count=3)
    # Solve renderable leaf material parameters that produce target reflectance and transmittance
    SI.solve_leaf_material_parameters(
        slab_sim_name=set_name, solver="nn", clear_old_results=True
    )
    # After solver has run, check results from HyperBlend\leaf_measurement_sets\try_random_p_leaves\set_result

    sampling = [450, 550, 600.0, 650, 700, 750, 800.1, 900.0]
    TH.write_sampling(set_name, sampling=sampling)
    sampling = TH.read_sampling(set_name)
    print(sampling)

    SI.resample_leaf_targets(set_name=set_name)
    plotter.plot_resampling(set_name=set_name)

    # Similarly, we can provide exact parameters
    set_name = "try_p_leaves"
    # generates a leaf target with certain parameters to \HyperBlend\leaf_measurement_sets\try_p_leaves\sample_targets.
    # The values used here are the default values.
    SI.generate_prospect_leaf(
        set_name=set_name,
        sample_id=0,
        n=1.5,
        ab=32,
        ar=8,
        brown=0,
        w=0.016,
        m=0.009,
        ant=0,
    )
    # You can also give only some parameters. Defaults will be used for the ones not provided.
    # Remember to give new sample_id so that the previously created leaf is not overwritten.
    SI.generate_prospect_leaf(set_name=set_name, sample_id=1, w=0.001, m=0.03)
    # Solve renderable leaf material parameters as before
    SI.solve_leaf_material_parameters(
        slab_sim_name=set_name, resolution=10, solver="nn"
    )
    # After solver has run, check results from HyperBlend\leaf_measurement_sets\try_p_leaves\set_result

    # We can also copy existing set and solve it with a different solver for example. Let's try that with
    #   surface fitting solver called 'surf'
    copy_set = "try_copying_set"
    SI.solve_leaf_material_parameters(
        slab_sim_name=copy_set, resolution=10, solver="surf", copyof="try_p_leaves"
    )

    # Let's try manually creating some data to work with
    set_name = "try_manual_set"
    # Example data list of lists where inner list holds the data ordered as [wavelength, reflectance, transmittance]
    data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
    # Write data to disk in a format the HyperBlend can understand
    TH.write_target(set_name, data, signal_id=0, resampled=False)
    # Solve as before
    SI.solve_leaf_material_parameters(
        slab_sim_name=set_name, resolution=1, solver="opt", clear_old_results=True
    )
