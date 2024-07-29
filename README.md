# The Better Regulations Tool

[![QA](https://github.com/DataS-DHSC//better-regulations-tool/actions/workflows/qa.yml/badge.svg)](https://github.com/DataS-DHSC//better-regulations-toolactions/workflows/qa.yml)

A tool to support the identification of Review Clauses in legislation.

## Setting up

In order to use this tool, you will need a machine with `conda` installed.

### Quickstart

To set up, open a terminal and create an environment from the provided environemnt.yml file using:
```
 conda env create -f environment.yml`
```
Then activate the environment using:
```
conda activate better-regulations-tool-env
```

To run the tool using the default settings use 

```
python search_main.py
```

To change the settings and a run a customized search edit the `example_config.yml` file.
This file is found in the `inputs/configs` folder
The parameters in the config.yml are explained in more detail in the user guide [here](https://github.com/DataS-DHSC/better-regulations-tool/blob/main/docs/user_guide.md)


## License

Unless stated otherwise, the codebase is released under the MIT License. This covers both the codebase and any sample code in the documentation. The documentation is © Crown copyright and available under the terms of the Open Government 3.0 licence.


### Requirements

-   conda installed

-   Python 3.12+ installed


## Project Organization

    ├── .github                 <- Github actions and workflows.
    ├── docs                    <- Markdown documentation and user guide.
    ├── inputs
    │   ├── data                    <- Raw and intermediary data.
    │   └── configs                 <- Config files.
    ├── outputs                 <- Output data, logs and configs.
    ├── src                     <- Source code.
    │   └── toolkit                 <- Toolkit subpackage.
    ├── tests                   <- Unit tests.
    ├── .flake8                 <- Linter configuration file for QA. 
    ├── .gitignore              <- Detailing folders and file types which will not be committed.
    ├── environment.yml         <- Dependency list for creating conda environment.
    ├── LICENSE                 <- License file.
    ├── main.py                 <- Main file for running tool.
    |── README.md               <- The top-level README.
    └── requirements.txt        <- Dependency list for QA only. Please use environment.yml 

### Contact

For queries, help or feedback please contact the  [DHSC Data Science Hub](mailto:datascience@dhsc.gov.uk).

