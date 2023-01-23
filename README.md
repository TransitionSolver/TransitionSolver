# How to run
Two example scripts are provided: `PipelineExample.py` and `BarebonesExample.py`. These can be run from the terminal but accept no arguments. `PipelineExample.py` has two example functions that perform the same task in slightly different ways: `example` and `example_parameterPointFile`. See their respective documentation for details.

# Defining a model
Unfortunately, defining a model currently requires double effort: it must be defined in `TransitionSolver` and `PhaseTracer`. In `PhaseTracer`, the model should extend either `Potential` or `OneLoopPotential`. In `TransitionSolver`, the model should extend `AnalysablePotential`, which in turn extends `CosmoTransitions`' `generic_potential`. See `ToyModel.hpp` in `PhaseTracer\EffectivePotential\include\models` and `ToyModel.py` in `TransitionSolver` for a simple example model.

# Global events
Simply import NotifyHandler into a script to enable a global event handler. Some classes query NotifyHandler to see if any actions should be taken when an event occurs. For example, ActionSampler (found in TransitionAnalysis.py) queries NotifyHandler when an instance of ActionSampler is created, and passes a reference of the instance to the NotifyHandler for handling. One could register the on_create event with NotifyHandler by calling

	import NotifyHandler
	import TransitionAnalysis
	def configureActionSampler(actionSampler: TransitionAnalysis.ActionSampler):
		actionSampler.bDebug = True
	notifyHandler.addEvent('ActionSampler-on_create', configureActionSampler)

Then, when ActionSampler is created, the configureActionSampler function will be called. This allows for very convenient configuration of objects that are not created directly from the main script. Note that the existing queried events currently all have a tag of the form `<className>-<eventName>`, and the event queried from the creation of an object always has the `eventName` of `on_create`. This is just a convention, and all other events can be named arbitrarily.

# Compatibility

## Operating system

`TransitionSolver` is written in Python, so it should be cross-platform. However, the example scripts will only work in Windows, specifically a version with Windows Subsystem for Linux. I believe all that needs to change to run the example scripts on Linux is the removal of the 'wsl' argument in suprocess.call. Compatibility of these example scripts on macOS is unknown.

## Packages and modules

The following packages are required for `TransitionSolver`:

* A modified version of `CosmoTransitions` that fixes many numerical errors.
* `PhaseTracer` in the `TransitionSolverInterface` branch.

The following non-core Python modules are required for `TransitionSolver`:

* `matplotlib`
* `numpy`
* `scipy`

# Support
For assistance, please contact: [lachlan.morris@monash.edu](mailto:lachlan.morris@monash.edu).
