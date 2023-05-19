# Requirements and compatibility

## Operating system

`TransitionSolver` is written in Python, so it should be cross-platform. However, the example scripts will only work in Windows, specifically a version with Windows Subsystem for Linux. I believe all that needs to change to run the example scripts on Linux is the removal of the `'wsl'` command in suprocess.call. Compatibility of these example scripts on macOS is unknown.

## Languages

Python 3.9 was used to develop `TransitionSolver`. At the very least, Python 3.5 is required for type hinting, however a later version may be required due to some other feature used in the program. The oldest compatible version is currently unknown.

I believe `PhaseTracer` requires at least C++11.

## Packages and modules

The following packages are required for `TransitionSolver`:

* A modified version of `CosmoTransitions` that fixes many numerical errors.
* `PhaseTracer` in the `TransitionSolverInterface` branch.

The following non-core Python modules are required for `TransitionSolver`:

* `matplotlib`
* `numpy`
* `scipy`

See [here](https://github.com/PhaseTracer/PhaseTracer#requirements) for the requirements for running `PhaseTracer`.

# How to run
Note: All scripts should be run from the root directory: `TransitionSolver/`.

Two example scripts are provided in the `examples` subdirectory: `PipelineExample.py` and `BarebonesExample.py`. These examples do not support any arguments from the command line, so modification must be done within the code. These examples can be run from the terminal but accept no arguments. `PipelineExample.py` has two example functions that perform the same task in slightly different ways: `example` and `example_parameterPointFile`. See their respective documentation for details. To run, use the commands:

	python -m examples.PipelineExample
	python -m examples.BarebonesExample

There is now a new way to run `TransitionSolver`, using `CommandLineInterface.py`, again in the `examples` subdirectory. This accepts arguments from the command line. Two methods for using this script are:

	python -m examples.CommandLineInterface <modelLabel> <outputFolderName> <inputFileName>
	python -m examples.CommandLineInterface <modelLabel> <outputFolderName> <parameter value 1> <parameter value 2> ... <parameter value n>
	
The first method reads parameter values from an input text file `<inputFileName>`. It must be a `.txt` file. The second method reads parameter values 1 to n from the command line. Both methods save results in the folder specified by `<outputFolderName>`. The argument `<modelLabel>` specifies which model to use. Currently supported model labels are `rss` for the real scalar singlet model, `rss_ht` for the high temperature expansion, and `toy` for the toy model. Here are some examples that can be run using the first method:

	python -m examples.CommandLineInterface rss output/RSS/RSS_BP<n> input/RSS/RSS_BP<n>.txt
	python -m examples.CommandLineInterface rss_ht output/RSS/RSS_BP1 input/RSS/RSS_BP1.txt
	python -m examples.CommandLineInterface toy output/Toy/Toy_BP<n> input/Toy/Toy_BP<n>.txt

Here, `<n>` ranges from 1 to 5 because only five benchmarks for the `rss` and `toy` models have been provided in the `input` subdirectory. The `rss_ht` model currently only has one benchmark. Equivantly, using the second method for running `CommandLineInterface`, one could do e.g.

	python -m examples.CommandLineInterface toy output/Toy/Toy_BP5 0.1040047755 250 3.5 0.2

# Defining a model
Unfortunately, defining a model currently requires double effort: it must be defined in `TransitionSolver` and `PhaseTracer`. In `PhaseTracer`, the model should extend either `Potential` or `OneLoopPotential`. In `TransitionSolver`, the model should extend `AnalysablePotential`, which in turn extends `CosmoTransitions`' `generic_potential`. See `ToyModel.hpp` in `PhaseTracer/EffectivePotential/include/models` and `ToyModel.py` in `TransitionSolver` for a simple example model.

# Global events
Simply import `NotifyHandler` into a script to enable a global event handler. Some classes query `NotifyHandler` to see if any actions should be taken when an event occurs. For example, `ActionSampler` (found in `TransitionAnalysis.py`) queries `NotifyHandler` when an instance of `ActionSampler` is created, and passes a reference of the instance to the `NotifyHandler` for handling. One could register the on_create event with `NotifyHandler` by calling

	import NotifyHandler
	import TransitionAnalysis
	def configureActionSampler(actionSampler: TransitionAnalysis.ActionSampler):
		actionSampler.bDebug = True
	notifyHandler.addEvent('ActionSampler-on_create', configureActionSampler)

Then, when `ActionSampler` is created, the `configureActionSampler` function will be called. This allows for very convenient configuration of objects that are not created directly from the main script. Note that the existing queried events currently all have a tag of the form `<className>-<eventName>`, and the event queried from the creation of an object always has the `eventName` of `on_create`. This is just a convention, and all other events can be named arbitrarily.

# Support
For assistance, please contact: [lachlan.morris@monash.edu](mailto:lachlan.morris@monash.edu).
