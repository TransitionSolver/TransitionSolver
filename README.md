# Requirements and compatibility

## Operating system

`TransitionSolver` is written in Python, so it should be cross-platform. The example scripts will work on Linux and can be made to work on Windows by following the relevant configuration step in the Configuration section below. MacOS has not yet been tested.

## Languages

Python 3.9 was used to develop `TransitionSolver` but we also support versions 3.7 and 3.8  by including including `from __future__ import annotations` at the top of relevant files.

The dependent package `PhaseTracer` requires at least a C++11 compatible compiler, see `PhaseTracer` documentation for more specifications.

## Packages and modules

The following packages are required for `TransitionSolver`:

* `CosmoTransitions` which is modified with the patch distributed with TransitionSolver util/CosmoTransitions.patch.  On linux this patch can be applied by copying it to the base directory of CosmoTransitions and doing:

$ patch -p1  < CosmoTransitions.patch

This  patch fixes a number of numerical errors that would otherwise disrupt the calculations performed by TransitionSolver.

* `PhaseTracer` in the `TransitionSolverInterface` branch for which you will also need git version control.  On linux OS you can try the simple script util/Install_PhaseTracer distributed with TransitionSolver. 

The following non-core Python modules are required for `TransitionSolver`:

* `matplotlib`
* `numpy`
* `scipy`

See [here](https://github.com/PhaseTracer/PhaseTracer#requirements) for the requirements for running `PhaseTracer`.

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

Two example scripts are provided in the `examples` subdirectory: `pipeline_example.py` and `barebones_example.py`. These examples do not support any arguments from the command line, so modification must be done within the code. These examples can be run from the terminal but accept no arguments. `pipeline_example.py` has two example functions that perform the same task in slightly different ways: `example` and `example_parameterPointFile`. See their respective documentation for details. To run, use the commands:

	python3 -m examples.pipeline_example
	python3 -m examples.barebones_example

For supported models `TransitionSolver` can also be run using `command_line_interface.py`, again in the `examples` subdirectory. This accepts arguments from the command line. Two methods for using this script are:

	python3 -m examples.command_line_interface <modelLabel> <outputFolderName> <inputFileName>
	python3 -m examples.command_line_interface <modelLabel> <outputFolderName> <parameter value 1> <parameter value 2> ... <parameter value n>
	
The first method reads parameter values from an input text file `<inputFileName>`. It must be a `.txt` file. The second method reads parameter values 1 to n from the command line. Both methods save results in the folder specified by `<outputFolderName>`. The argument `<modelLabel>` specifies which model to use. Currently supported model labels are `rss` for the real scalar singlet model, `rss_ht` for the high temperature expansion, and `toy` for the toy model. Here are some examples that can be run using the first method:

	python3 -m examples.command_line_interface rss output/RSS/RSS_BP<n> input/RSS/RSS_BP<n>.txt
	python3 -m examples.command_line_interface rss_ht output/RSS_HT/RSS_HT_BP1 input/RSS_HT/RSS_HT_BP1.txt
	python3 -m examples.command_line_interface toy output/Toy/Toy_BP<n> input/Toy/Toy_BP<n>.txt

Here, `<n>` ranges from 1 to 5 because only five benchmarks for the `rss` and `toy` models have been provided in the `input` subdirectory. The `rss_ht` model currently only has one benchmark. Equivantly, using the second method for running `command_line_interface`, one could do e.g.

	python3 -m examples.command_line_interface toy output/Toy/Toy_BP5 0.1040047755 250 3.5 0.2

# Computing gravitational waves
TransitionSolver also comes woth a module for computing the GW spectrum.  The GW spectra are computed from fit formulae as described in [arXiv:2309.05474](https://arxiv.org/abs/2309.05474) and [arXiv:2306.17239](https://arxiv.org/abs/2306.17239).  To use the GW spectrum for a point after Transition has already been run
using one of the options above please do

      python3 -m gws.gw_analyser 

PA: TODO change code to take arguements for model and TSOutputFolder and GWsOutputFolder and to choose between using scanGWsWithParam and scanGWs



# Defining a model
Unfortunately, defining a model currently requires double effort: it must be defined in `TransitionSolver` and `PhaseTracer`. In `PhaseTracer`, the model should extend either `Potential` or `OneLoopPotential`. In `TransitionSolver`, the model should extend `AnalysablePotential`, which in turn extends `CosmoTransitions`' `generic_potential`. See `ToyModel.hpp` in `PhaseTracer/EffectivePotential/include/models` and `toy_model.py` in `TransitionSolver` for a simple example model.  In future versions it should be possible to only enter one potential and use PhaseTracer to compute the bounce action. 

# Global events
Simply import `util.events.NotifyHandler` into a script to enable a global event handler. Some classes query `NotifyHandler` to see if any actions should be taken when an event occurs. For example, `ActionSampler` (found in `transition_analysis.py`) queries `NotifyHandler` when an instance of `ActionSampler` is created, and passes a reference of the instance to the `NotifyHandler` for handling. One could register the on_create event with `NotifyHandler` by calling

	from util.events import notifyHandler
	from analysis.transition_analysis import ActionSampler
	def configureActionSampler(actionSampler: ActionSampler):
		actionSampler.bDebug = True
	notifyHandler.addEvent('ActionSampler-on_create', configureActionSampler)

Then, when `ActionSampler` is created, the `configureActionSampler` function will be called. This allows for very convenient configuration of objects that are not created directly from the main script. Note that the existing queried events currently all have a tag of the form `<className>-<eventName>`, and the event queried from the creation of an object always has the `eventName` of `on_create`. This is just a convention, and all other events can be named arbitrarily.

# Support
For assistance, please contact: [lachlan.morris@monash.edu](mailto:lachlan.morris@monash.edu).
