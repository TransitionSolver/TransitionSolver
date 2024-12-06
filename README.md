# Warning

TransitiionSolver 1.0 has not yet been released. If you are reading this you are looking at an in development version of the code. The code has been used in physics projects and performs rigorous state of the art calculations for phase transitions and GWs.  However it currently lacks complete documentation, may lack some user friendliness and require some adjustments to make it work for your own case, or even adaption of sampling algorithms (these are currently being generalised in another branch). Furthermore the caveats of what it can and can not handle may be unclear.

# Requirements and compatibility

## Operating system

`TransitionSolver` is written in Python, so it should be cross-platform. The example scripts will work on Linux and can be made to work on Windows by following the relevant configuration step in the Configuration section below. MacOS has not yet been tested.

## Languages

Python 3.9 was used to develop `TransitionSolver` but we also support versions 3.7 and 3.8  by including including `from __future__ import annotations` at the top of relevant files.

The dependent package `PhaseTracer` requires at least a C++11 compatible compiler, see `PhaseTracer` documentation for more specifications.

## Packages and modules

The following packages are required for `TransitionSolver`:

The following non-core Python modules are required for `TransitionSolver`:

* `matplotlib`
* `numpy`
* `scipy`

* `PhaseTracer`.  This can be installed using

$ git clone https://github.com/PhaseTracer/PhaseTracer PhaseTracer
$ cd PhaseTracer
$ mkdir build
$ cmake ..
$ make -j12

See [here](https://github.com/PhaseTracer/PhaseTracer#requirements) for the requirements for running `PhaseTracer`.

* `CosmoTransitions` which is modified with the patch distributed with TransitionSolver util/CosmoTransitions.patch.  On linux this patch can be applied by copying it to the base directory of CosmoTransitions and doing:

$ patch -p1  < CosmoTransitions.patch

This  patch fixes a number of numerical errors that would otherwise disrupt the calculations performed by TransitionSolver.  In future versions we expect the CosmoTransitions Dependency will be removed by using PhaseTracer for the bounce action.

# Configuration
Currently only user-specific settings are configurable. From the root directory, create a configuration folder and file: `config/config_user.json`. Add:

	{
		"PhaseTracer_directory": "<location of PhaseTracer>"
	}
	
to the configuration file. The location of `PhaseTracer` can be specified as a relative or absolute path. If it is a relative path, it should be relative to the root directory: `TransitionSolver/`.

If you are using Windows, `PhaseTracer` must be run using WSL. This requires adding the setting `"Windows": true` to `config_user.json`, which would then read:

	{
		"PhaseTracer_directory": "<location of PhaseTracer>",
		"Windows": true
	}
	
The `Windows` entry could be set to false if you are not using Windows, or can be omitted entirely.

# How to run
Note: All scripts should be run from the root directory: `TransitionSolver/`. Before running any scripts, `TransitionSolver` must first be configured for the machine it is run on --- see the Configuration section.

Two example scripts are provided in the `examples` subdirectory: `pipeline_example.py` and `barebones_example.py`. These examples These examples can be run from the terminal, but accept no arguments from the command line, so modification must be done within the code.  `pipeline_example.py` has two example functions that perform the same task in slightly different ways: `example` and `example_parameterPointFile`. See their respective documentation for details. To run, use the commands:

	python3 -m examples.pipeline_example
	python3 -m examples.barebones_example

For supported models `TransitionSolver` can also be run using `command_line_interface.py`, again in the `examples` subdirectory. This accepts arguments from the command line. Two methods for using this script are:

	python3 -m examples.command_line_interface <modelLabel> <GWs> <outputFolderName> <inputFileName>
	python3 -m examples.command_line_interface <modelLabel> <GWs> <outputFolderName> <parameter value 1> <parameter value 2> ... <parameter value n>
	
The first method reads parameter values from an input text file `<inputFileName>`. It must be a `.txt` file. The second method reads parameter values 1 to n from the command line. Both methods save results in the folder specified by `<outputFolderName>`. The argument `<modelLabel>` specifies which model to use. Currently supported model labels are `rss` for the real scalar singlet model, `rss_ht` for the high temperature expansion, `toy` for the toy model and 'smpluscubic' for a model based upon a non-linear realisation of EWSB that can exibit strong supercooling.

The second argument `<GWs>` should be an integer that specifies whether or not to use output of the basic functionality TransitionSolver to compute graviational waves (GWs),  and if yes with what sources.  0 means no GWs are computed, 1 means GWs with fluid contributions (sound waves and turbulence) only, 2 means gravitational waves from the scalar field / bubble collisions only, and 3 we present separate GWs predictions for bubble collisions and fluid contributions but we do not combine them. See Computing gravitational waves section below for more details.

Here are some examples that can be run using the first method where the input file is passed:

	python3 -m examples.command_line_interface rss 1 output/RSS/RSS_BP<n> input/RSS/RSS_BP<n>.txt
	python3 -m examples.command_line_interface rss_ht 1 output/RSS_HT/RSS_HT_BP1 input/RSS_HT/RSS_HT_BP1.txt
	python3 -m examples.command_line_interface toy 1 output/Toy/Toy_BP<n> input/Toy/Toy_BP<n>.txt

Here, `<n>` ranges from 1 to 5 because only five benchmarks for the `rss` and `toy` models have been provided in the `input` subdirectory. The `rss_ht` model currently only has one benchmark. Equivalently, using the second method for running `command_line_interface`, one could do e.g.\

	python3 -m examples.command_line_interface toy 0 output/Toy/Toy_BP5 0.1040047755 250 3.5 0.2

Note that for the bubble wall velocity the user may either choose to fix vw to the Chapman-Jouguet velocity which ensures it transforms as a detonation for which the TransitionSolver calculations code is most reliable or to a value input by the user.  Currently this is controlled by the transitionAnalyser boolean flag bUseChapmanJouguetVelocity and  GWAnalysisSetting  flag of same name. In the command line interface these are both controled by the bool bUseCJvw.  The input vw is passed as an argument of analysePhaseHistory_supplied which is called in all the example scripts and the command line interface.  However be warned that fro GWs choose a vw that is smaller than the Chapman Jouguet velocity will lead to an error being thrown as currenly only this is supported.  This will be chnaged in the near future.   

# Computing gravitational waves

TransitionSolver also comes with a module for computing the GW spectrum in the gws folder.  The GW spectra are computed from fit formulae as described in [arXiv:2309.05474](https://arxiv.org/abs/2309.05474) and [arXiv:2306.17239](https://arxiv.org/abs/2306.17239).  To compute gravitational waves while running basic functionality of PhaseTracer please use the command line arguments above, specifying 1, 2 or 3 for GWs. If GWs is set to:

1  the GWs are computed only from fluid contributions (sound waves and turbulence) where we assume friction leads to all the energy in the scalar field being transferred to the fluid.  TransitionSolver has mostly been developed with this in mind. The efficiency coefficient kappa_{turb} = 0.05 by default but could be modified by changing the settings object from class GWAnalysisSettings that is passed in the command_line_interface.py script.

2 the GWs are computed for only the bubble collisions  / scalar field source, where we assume a runaway bubble wall due to a lack of friction means all the energy remains in the scalar field.  For this source we assume the efficiency coefficient $\kappa_{coll} = 1$, though this can also be set by a user if they change the settings object passed in the command_line_interface.py.

3  GWs predictions from both fluid only and from bubble collisions only are computed and presented separately.  



To use the GW spectrum for a point after Transition has already been run and the results saved to an output folder <TS_Output_Directory> then you can compute (or recompute) the GWs spectrum and create plots showing how the results depend on the transition temperature (similar to plats that appear in https://arxiv.org/abs/2309.05474.  To do this one can use the following command:

      python3 -m gws.gw_analyser <modelname> <GWsPlotsDir>  <TS_Output_Directory>  

If no command line arguments are specified, ie

      python3 -m gws.gw_analyser

then a default model and location will be used based on defaults that are specified in gw_analyser.py (search for default_model).  By default the signal to noise ratio is computed for LISA.  However if there is an alternative class added in gws/detectors (see the gws/detectors/lisa.py for an example) then the name of this class can be passed as a final optional argument.  

Note if you use it this way, you must currently manually ensure that you set the bUseChapmanJouguetVelocity in the main routine of the gw_analyser script, to the same value as used in the original run of TransitionSolver.

Running this command takes order O(5 minutes) on a single core, as it computes at many different transiton temperatures. To compute the GW spectrum only at the percolation temperature for which the final SNR (signal to noise ratio)  is output to terminal, you can comment out the line

gwa.scanGWs(GWsOutputFolder, bCombined=False)

in the main routine of the gw_analyser module.  This should run in order O(1 second).

# Defining a model
Unfortunately, defining a model currently requires double effort: it must be defined in `TransitionSolver` and `PhaseTracer`. In `PhaseTracer`, the model should extend either `Potential` or `OneLoopPotential`. In `TransitionSolver`, the model should extend `AnalysablePotential`, which in turn extends `CosmoTransitions`' `generic_potential`. See `ToyModel.hpp` in `PhaseTracer/EffectivePotential/include/models` and `toy_model.py` in `TransitionSolver` for a simple example model.  In future versions it should be possible to only enter one potential and use PhaseTracer to compute the bounce action.  In future versions this duplication of effort should be removed.

# Global events
Simply import `util.events.NotifyHandler` into a script to enable a global event handler. Some classes query `NotifyHandler` to see if any actions should be taken when an event occurs. For example, `ActionSampler` (found in `transition_analysis.py`) queries `NotifyHandler` when an instance of `ActionSampler` is created, and passes a reference of the instance to the `NotifyHandler` for handling. One could register the on_create event with `NotifyHandler` by calling

	from util.events import notifyHandler
	from analysis.transition_analysis import ActionSampler
	def configureActionSampler(actionSampler: ActionSampler):
		actionSampler.bDebug = True
	notifyHandler.addEvent('ActionSampler-on_create', configureActionSampler)

Then, when `ActionSampler` is created, the `configureActionSampler` function will be called. This allows for very convenient configuration of objects that are not created directly from the main script. Note that the existing queried events currently all have a tag of the form `<className>-<eventName>`, and the event queried from the creation of an object always has the `eventName` of `on_create`. This is just a convention, and all other events can be named arbitrarily.

# Support
Until release of the full TransitionSolver manual, limited documentation and support is available.  However if you encouter problems you may contact [transitionsolver@gmail.com](mailto:transitionsolver@gmail.com) with queries or bug reports. 
