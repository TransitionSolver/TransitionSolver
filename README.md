<h1 align="center">
TransitionSolver
</h1>

<div align="center">
<i>Solve properties of first-order phase transitions and predict their gravitational wave spectrum in scalar-field theories</i>
</div>
<br>
<div align="center">
<img alt="GitHub License" src="https://img.shields.io/github/license/TransitionSolver/TransitionSolver">
</div>
<br>

**TransitionSolver** is a Python3 software package for solving properties of cosmological phase transitions and plotting the gravitational wave spectrum for Standard Model extensions with any number of scalar fields.

# Installation 

Install `TransitionSolver` by

```bash
pipx install git+https://github.com/TransitionSolver/TransitionSolver
```
This may require e.g.
```
apt install python3-pipx
```

Running `TransitionSolver` requires `PhaseTracer`, which can be installed by
```bash
# install PhaseTracer and dependencies on debian-based system e.g. Ubuntu
wget https://raw.githubusercontent.com/TransitionSolver/TransitionSolver/refs/heads/main/debian_installer_phasetracer.bash -O - | sudo bash 
# or handle installation yourself and do
export PHASETRACER=/path/to/your/phasetracer/
```
See [here](https://github.com/PhaseTracer/PhaseTracer#requirements) for further installation instructions and the requirements for running `PhaseTracer`.

# Command line interface

TransitonSolver can be ran at the command-line. See:

```bash
$ ts --help
Usage: ts [OPTIONS]

  Run TransitionSolver on a particular model and point

  Example usage:

  ts --model RSS --point input/RSS/RSS_BP1.txt --action-ct --apply set_daisy_method 2 --apply set_bUseBoltzmannSuppression True --force

Options:
  --model TEXT                    Model name  [required]
  --model-header PATH             Model header-file
  --model-lib PATH                Library for model if not header-only
  --model-namespace TEXT          Namespace for model
  --point PATH                    Parameter point file  [required]
  --vw FLOAT RANGE                Bubble wall velocity  [x>=0.0]
  --detector [LISA|LISA_SNR_10]   Gravitational wave detector
  --pta [NANOGrav|PPTA|EPTA]      Pulsar Timing Array
  --show BOOLEAN                  Whether to show plots
  --level [debug|info|warning|error|critical]
                                  Logging level
  --apply <TEXT LITERAL_EVAL>...  Apply settings to a potential
  --force                         Force recompilation
  --action-ct                     Use CosmoTransitions for action
  --help                          Show this message and exit.
```
For example, try
```bash
ts --model RSS --point ./tests/rss_bp1.txt 
```
You can pass a model and model header file etc, and parameter point.

# Scans

# Defining a model

`TransitionSolver` uses models written in C++ in `PhaseTracer`. A new model should implment either a `Potential` or `OneLoopPotential` base class.

# Credit and citations

The authors of `TransitionSolver` are Andrew Fowlie, Peter Athron, Csaba Balazs, Lachlan Morris, with the bulk of the original code having been developed by Lachlan Morris during his PhD. If you use `TransitionSolver`, please cite

```bibtex
@article{Athron:2022mmm,
    author = "Athron, Peter and Bal{\'a}zs, Csaba and Morris, Lachlan",
    title = "{Supercool subtleties of cosmological phase transitions}",
    eprint = "2212.07559",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1088/1475-7516/2023/03/006",
    journal = "JCAP",
    volume = "03",
    pages = "006",
    year = "2023"
}

@article{Athron:2024xrh,
    author = "Athron, Peter and Balazs, Csaba and Fowlie, Andrew and Morris, Lachlan and Searle, William and Xiao, Yang and Zhang, Yang",
    title = "{PhaseTracer2: from the effective potential to gravitational waves}",
    eprint = "2412.04881",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1140/epjc/s10052-025-14258-y",
    journal = "Eur. Phys. J. C",
    volume = "85",
    number = "5",
    pages = "559",
    year = "2025"
}

@article{Athron:2020sbe,
    author = "Athron, Peter and Bal\'azs, Csaba and Fowlie, Andrew and Zhang, Yang",
    title = "{PhaseTracer: tracing cosmological phases and calculating transition properties}",
    eprint = "2003.02859",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "CoEPP-MN-20-3",
    doi = "10.1140/epjc/s10052-020-8035-2",
    journal = "Eur. Phys. J. C",
    volume = "80",
    number = "6",
    pages = "567",
    year = "2020"
}
```
