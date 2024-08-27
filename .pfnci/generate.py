#!/usr/bin/env python3

import argparse
import json
import os
import shlex  # requires Python 3.8
import sys

import yaml

from typing import Any, Dict, List, Mapping, Tuple


SchemaType = Mapping[str, Any]


class Matrix:
    def __init__(self, record: Mapping[str, Any]):
        """"""

        self._rec = {
            '_inherits': None,
            '_extern': False,
        }
        self._rec.update(record)

    def env(self) -> Dict[str, Any]:
        envvars = {}
        for k, v in self._rec.items():
        """"""

           if not k.startswith('env:') or v is None:
                continue
            envvars[k.split(':', 2)[1]] = v
        return envvars

    def __getattr__(self, key: str) -> Any:
        """"""

        if key in self._rec:
            return self._rec[key]
        raise AttributeError(f'"{key}" not defined in matrix {self._rec}')

    def copy(self) -> 'Matrix':
        """"""

        return Matrix(self._rec.copy())

    def update(self, matrix: 'Matrix') -> None:
        """Updates the matrix with the values from another matrix.
        Parameters:
            - matrix (Matrix): The matrix to update with.
        Returns:
            - None: The function does not return anything.
        Processing Logic:
            - Updates matrix with values from another.
            - Uses self._rec and matrix._rec.
            - _rec is a dictionary.
            - _rec is a dictionary."""

        self._rec.update(matrix._rec)


class LinuxGenerator:
    def __init__(self, schema: SchemaType, matrix: Matrix):
        """"Initializes a new instance of the class with the given schema and matrix.
        Parameters:
            - schema (SchemaType): The schema to be used for the matrix.
            - matrix (Matrix): The matrix to be used for the instance.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Asserts that the matrix system is 'linux'.
            - Sets the schema and matrix attributes for the instance.""""

        assert matrix.system == 'linux'
        self.schema = schema
        self.matrix = matrix

    def generate_dockerfile(self) -> str:
        """Function:
        def generate_dockerfile(self) -> str:
            Generates a Dockerfile based on the provided matrix parameters.
            Parameters:
                - self (object): The current object.
            Returns:
                - str: The generated Dockerfile as a string.
            Processing Logic:
                - Generates a base image based on the provided matrix parameters.
                - Installs necessary tools and libraries.
                - Sets environment variables for ROCm.
                - Sets up Python and its libraries.
                - Uninstalls any unnecessary libraries."""

        matrix = self.matrix
        lines = [
            '# AUTO GENERATED: DO NOT EDIT!',
        ]

        os_name, os_version = matrix.os.split(':')
        if matrix.cuda is not None:
            full_ver = self.schema['cuda'][matrix.cuda]['full_version']
            base_image = f'nvidia/cuda:{full_ver}-devel-{os_name}{os_version}'
        elif matrix.rocm is not None:
            full_ver = self.schema['rocm'][matrix.rocm]['full_version']
            base_image = f'rocm/dev-{os_name}-{os_version}:{full_ver}'
        else:
            raise AssertionError

        lines += [
            f'ARG BASE_IMAGE="{base_image}"',
            'FROM ${BASE_IMAGE}',
            '',
        ]

        # Install tools and additional libraries.
        if os_name == 'ubuntu':
            if matrix.rocm is not None:
                # GPG key has expired in ROCm 4.2 (or earlier) docker images
                lines += [
                    'RUN export DEBIAN_FRONTEND=noninteractive && \\',
                    '    ( apt-get -qqy update || true ) && \\',
                    '    apt-get -qqy install ca-certificates && \\',
                    '    curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -',  # NOQA
                ]
            elif matrix.cudnn is not None:
                major = matrix.cudnn.split('.')[0]
                if major == '7':
                    ubuntu_version = os_version.replace('.', '')
                    lines += [
                        'RUN export DEBIAN_FRONTEND=noninteractive && \\',
                        '    apt-get -qqy update && \\',
                        '    apt-get -qqy install software-properties-common && \\',  # NOQA
                        f'    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu{ubuntu_version}/x86_64/7fa2af80.pub && \\',  # NOQA
                        f'    add-apt-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu{ubuntu_version}/x86_64/ /"',  # NOQA
                        '']

            lines += [
                'RUN export DEBIAN_FRONTEND=noninteractive && \\',
                '    apt-get -qqy update && \\',

                # pyenv: https://github.com/pyenv/pyenv/wiki
                '    apt-get -qqy install \\',
                '       make build-essential libssl-dev zlib1g-dev \\',
                '       libbz2-dev libreadline-dev libsqlite3-dev wget \\',
                '       curl llvm libncursesw5-dev xz-utils tk-dev \\',
                '       libxml2-dev libxmlsec1-dev libffi-dev \\',
                '       liblzma-dev \\',
                '       libopenmpi-dev \\' if matrix.mpi4py else '\\',
                '       && \\',
                '    apt-get -qqy install ccache git curl && \\',
                '    apt-get -qqy --allow-change-held-packages \\',
                '            --allow-downgrades install {}'.format(
                    shlex.join(self._additional_packages('apt'))
                ),
                '',
                'ENV PATH "/usr/lib/ccache:${PATH}"',
                '',
            ]
        elif os_name == 'centos':
            assert os_version in ('7', '8')
            if os_version == '7':
                lines += [
                    'RUN yum -y install centos-release-scl && \\',
                    '    yum -y install devtoolset-7-gcc-c++',
                    'ENV PATH "/opt/rh/devtoolset-7/root/usr/bin:${PATH}"',
                    'ENV LD_LIBRARY_PATH "/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:${LD_LIBRARY_PATH}"',  # NOQA
                    '',
                ]

            lines += [
                # pyenv: https://github.com/pyenv/pyenv/wiki
                'RUN yum -y install \\',
                '       zlib-devel bzip2 bzip2-devel readline-devel sqlite \\',
                '       sqlite-devel openssl-devel tk-devel libffi-devel \\',
                '       xz-devel && \\',
                '    yum -y install epel-release && \\',
                '    yum -y install "@Development Tools" ccache git curl && \\',  # NOQA
                '    yum -y install {}'.format(
                    shlex.join(self._additional_packages('yum'))
                ),
                '',
                'ENV PATH "/usr/lib64/ccache:${PATH}"',
                '',
            ]
            assert matrix.mpi4py is None, 'mpi4py test unsupported on CentOS'
        else:
            raise AssertionError

        # Update alternatives for cuTENSOR for the current CUDA version.
        if matrix.cutensor is not None:
            lines += [
                'COPY setup/update-alternatives-cutensor.sh /',
                'RUN /update-alternatives-cutensor.sh',
                '',
            ]

        # Set environment variables for ROCm.
        if matrix.rocm is not None:
            lines += [
                'ENV ROCM_HOME "/opt/rocm"',
                'ENV LD_LIBRARY_PATH "${ROCM_HOME}/lib"',
                'ENV CPATH "${ROCM_HOME}/include"',
                'ENV LDFLAGS "-L${ROCM_HOME}/lib"',
                '',
            ]

        # Setup Python.
        if matrix.python is None:
            raise ValueError('Python cannot be null')

        py_spec = self.schema['python'][matrix.python]['spec']
        lines += [
            'RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv',
            'ENV PYENV_ROOT "/opt/pyenv"',
            'ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"',
            f'RUN pyenv install {py_spec} && \\',
            f'    pyenv global {py_spec} && \\',
            '    pip install -U setuptools pip wheel',
            '',
        ]

        # Setup Python libraries.
        pip_args = []
        pip_uninstall_args = []
        for pylib in ('numpy', 'scipy', 'optuna', 'mpi4py',
                      'cython', 'cuda-python'):
            pylib_ver = getattr(matrix, pylib)
            if pylib_ver is None:
                pip_uninstall_args.append(pylib)
            else:
                pip_spec = self.schema[pylib][pylib_ver]['spec']
                pip_args.append(f'{pylib}{pip_spec}')
        lines += [
            f'RUN pip install -U {shlex.join(pip_args)}',
        ]
        if len(pip_uninstall_args) != 0:
            # Ensure that packages are not installed.
            lines += [
                f'RUN pip uninstall -y {shlex.join(pip_uninstall_args)} && \\',
                '    pip check',
            ]
        lines.append('')
        return '\n'.join(lines)

    def _additional_packages(self, kind: str) -> List[str]:
        """('Unexpected code path')
        This function returns a list of additional packages required for a given kind of installation (apt or yum) based on the specified matrix parameters.
        Parameters:
            - kind (str): Specifies the kind of installation (apt or yum).
        Returns:
            - List[str]: A list of additional packages required for the specified kind of installation.
        Processing Logic:
            - Checks if the specified kind is either 'apt' or 'yum'.
            - If the specified matrix has a cuda parameter, it checks for additional packages related to nccl, cutensor, cusparselt, and cudnn.
            - If the specified matrix has a rocm parameter, it returns the list of packages specified in the schema.
            - If the specified matrix has neither a cuda nor a rocm parameter, it raises an AssertionError."""

        assert kind in ('apt', 'yum')
        matrix = self.matrix
        if matrix.cuda is not None:
            packages = []
            apt = kind == 'apt'
            cuda = matrix.cuda
            nccl = matrix.nccl
            cutensor = matrix.cutensor
            cusparselt = matrix.cusparselt
            cudnn = matrix.cudnn
            if nccl is not None:
                spec = self.schema['nccl'][nccl]['spec']
                major = nccl.split('.')[0]
                if apt:
                    packages.append(f'libnccl{major}={spec}+cuda{cuda}')
                    packages.append(f'libnccl-dev={spec}+cuda{cuda}')
                else:
                    packages.append(f'libnccl-{spec}-*+cuda{cuda}')
                    packages.append(f'libnccl-devel-{spec}-*+cuda{cuda}')
            if cutensor is not None:
                spec = self.schema['cutensor'][cutensor]['spec']
                major = cutensor.split('.')[0]
                if apt:
                    packages.append(f'libcutensor{major}={spec}')
                    packages.append(f'libcutensor-dev={spec}')
                else:
                    packages.append(f'libcutensor{major}-{spec}')
                    packages.append(f'libcutensor-devel-{spec}')
            if cusparselt is not None:
                spec = self.schema['cusparselt'][cusparselt]['spec']
                major = cusparselt.split('.')[0]
                if apt:
                    packages.append(f'libcusparselt{major}={spec}')
                    packages.append(f'libcusparselt-dev={spec}')
                else:
                    packages.append(f'libcusparselt{major}-{spec}')
                    packages.append(f'libcusparselt-devel-{spec}')
            if cudnn is not None:
                spec = self.schema['cudnn'][cudnn]['spec']
                cudnn_cuda_schema = self.schema['cudnn'][cudnn]['cuda'][cuda]
                alias = cuda
                if cudnn_cuda_schema is not None:
                    alias = cudnn_cuda_schema['alias']
                major = cudnn.split('.')[0]
                if apt:
                    packages.append(f'libcudnn{major}={spec}+cuda{alias}')
                    packages.append(f'libcudnn{major}-dev={spec}+cuda{alias}')
                else:
                    packages.append(
                        f'libcudnn{major}-{spec}-*.cuda{alias}')
                    packages.append(
                        f'libcudnn{major}-devel-{spec}-*.cuda{alias}')
            return packages
        elif matrix.rocm is not None:
            return self.schema['rocm'][matrix.rocm]['packages']  # type: ignore[no-any-return] # NOQA
        raise AssertionError

    def generate_script(self) -> str:
        """Function:
        def generate_script(self) -> str:
            Generates a bash script for running tests based on the provided matrix.
            Parameters:
                - matrix (Matrix): A matrix object containing information about the desired test configuration.
            Returns:
                - str: A bash script for running tests based on the provided matrix.
            Processing Logic:
                - Generates a list of lines for the bash script.
                - Adds environment variables to the list based on the matrix object.
                - Adds commands for building, testing, and cleaning up to the list.
                - Joins the list into a single string and returns it."""

        matrix = self.matrix
        lines = [
            '#!/bin/bash',
            '',
            '# AUTO GENERATED: DO NOT EDIT!',
            '',
            'set -uex',
            '',
            'ACTIONS="$(dirname $0)/actions"',
            '. "$ACTIONS/_environment.sh"',
            '',
        ]

        if matrix.cuda is not None:
            lines += [
                'export NVCC="ccache nvcc"',
                '',
            ]
        elif matrix.rocm is not None:
            lines += [
                '# TODO(kmaehashi): Tentatively sparsen parameterization to make test run complete.',  # NOQA
                'export CUPY_TEST_FULL_COMBINATION="0"',
                'export CUPY_INSTALL_USE_HIP=1',
                '',
            ]
        else:
            raise AssertionError

        for key, value in matrix.env().items():
            lines += [
                f'export {key}="{value}"',
                '',
            ]

        lines += ['"$ACTIONS/build.sh"']
        if matrix.test.startswith('unit'):
            if matrix.test == 'unit':
                spec = 'not slow and not multi_gpu'
            elif matrix.test == 'unit-multi':
                lines += [
                    'export OMPI_ALLOW_RUN_AS_ROOT=1',
                    'export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1'
                ]
                spec = 'not slow and multi_gpu'
            elif matrix.test == 'unit-slow':
                # Slow tests may use multiple GPUs.
                spec = 'slow'
            else:
                assert False
            lines += [f'"$ACTIONS/unittest.sh" "{spec}"']
        elif matrix.test == 'example':
            lines += ['"$ACTIONS/example.sh"']
        elif matrix.test == 'benchmark':
            lines += ['"$ACTIONS/benchmark.sh"']
        else:
            raise AssertionError

        lines += [
            '"$ACTIONS/cleanup.sh"',
            ''
        ]

        return '\n'.join(lines)


class CoverageGenerator:
    def __init__(self, schema: SchemaType, matrixes: List[Matrix]):
        """Initializes the object with the provided schema and list of matrixes.
        Parameters:
            - schema (SchemaType): The schema to be used for the matrixes.
            - matrixes (List[Matrix]): A list of matrixes to be used for processing.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Initialize object with schema and matrixes.
            - Assign schema to self.schema.
            - Assign matrixes to self.matrixes.
            - No additional processing is done."""

        self.schema = schema
        self.matrixes = matrixes

    def generate_rst(self) -> Tuple[str, List[str]]:
        """Generate_rst:
            Generates a matrix table for CuPy CI test coverage.
        Parameters:
            - self (object): The object that the function is called on.
        Returns:
            - Tuple[str, List[str]]: A tuple containing a string of reST output and a list of coverage warnings.
        Processing Logic:
            - Generates a matrix table.
            - Adds links to FlexCI projects.
            - Renders the matrix table as reST."""

        # Generate a matrix table.
        table = [
            ['Param', '', '#', 'Test'] + [''] * (len(self.matrixes) - 1),
            ['', 'System', ''] + [m.system for m in self.matrixes],
            ['', 'Target', ''] + [
                f'`{m.target} <t{i}_>`_ `🐳 <d{i}_>`_ `📜 <s{i}_>`_'
                for i, m in enumerate(self.matrixes)
            ],
            [''] * (len(self.matrixes) + 3)
        ]
        coverage_warns = []
        for key, key_schema in self.schema.items():
            possible_values = key_schema.keys()
            matrix_values = [getattr(m, key) for m in self.matrixes]
            key_header = key
            for value in possible_values:
                count = matrix_values.count(value)
                table += [
                    [
                        key_header,
                        value if value else 'null',
                        str(count) if count != 0 else '🚨',
                    ] + [
                        '✅' if mv == value else '' for mv in matrix_values
                    ],
                ]
                key_header = ''
                if count == 0:
                    coverage_warns.append(f'Uncovered axis: {key} = {value}')

        # Prepare reST output.
        lines = [
            '.. AUTO GENERATED: DO NOT EDIT!',
            '',
            'CuPy CI Test Coverage',
            '=====================',
            '',
            '.. list-table::',
            '   :header-rows: 3',
            '   :stub-columns: 2',
            '',
        ]

        # Render the matrix table as reST.
        for row in table:
            col0 = row[0]
            lines += [
                f'   * - {col0}',
            ] + [
                f'     - {col}' for col in row[1:]
            ]

        # Add links to FlexCI projects.
        lines += ['']
        for i, m in enumerate(self.matrixes):
            url = f'https://ci.preferred.jp/{m.project}/'
            if hasattr(m, '_url'):
                url = m._url
            lines += [
                f'.. _t{i}: {url}',
                f'.. _d{i}: {m.system}/tests/{m.target}.Dockerfile',
                f'.. _s{i}: {m.system}/tests/{m.target}.sh',
            ]
        lines += ['']

        return '\n'.join(lines), coverage_warns


class TagGenerator:
    def __init__(self, matrixes: List[Matrix]):
        """Initializes the object with a list of matrixes.
        Parameters:
            - matrixes (List[Matrix]): A list of matrixes to be stored in the object.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Initializes the object with a list of matrixes.
            - Stores the list of matrixes in the object.
            - No additional processing logic is applied.
            - This function does not modify the input parameters."""

        self.matrixes = matrixes

    def generate(self) -> str:
        """Function: Generate a JSON string containing project tags from a list of matrixes.
        Parameters:
            - self (type): An instance of the class.
        Returns:
            - str: A JSON string containing project tags.
        Processing Logic:
            - Loop through each matrix in the list.
            - Check if the matrix has tags.
            - Add the project and tags to the output dictionary.
            - Convert the output dictionary to a JSON string.
        Example:
            matrix1 = Matrix("Project A", ["tag1", "tag2"])
            matrix2 = Matrix("Project B", ["tag3", "tag4"])
            matrixes = [matrix1, matrix2]
            output = generate(matrixes)
            print(output)
            # Output: {
            #     "Project A": ["tag1", "tag2"],
            #     "Project B": ["tag3", "tag4"]
            # }"""

        output = {}
        for matrix in self.matrixes:
            if matrix.tags is not None:
                output[matrix.project] = matrix.tags
        return json.dumps(output, indent=4)


def validate_schema(schema: SchemaType) -> None:
    # Validate schema consistency
    for key, key_schema in schema.items():
        """This function validates the consistency of a given schema and raises a ValueError if any inconsistencies are found.
        Parameters:
            - schema (SchemaType): A dictionary representing the schema to be validated.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Validates the consistency of the given schema.
            - Raises a ValueError if any inconsistencies are found.
            - Checks for missing or unknown values for certain keys in the schema.
            - Only checks for specific keys and values, not the entire schema."""

        if key == 'os':
            for value, value_schema in key_schema.items():
                system = value_schema.get('system', None)
                if system is None:
                    raise ValueError(
                        f'system is missing '
                        f'while parsing schema os:{value}')
                if system not in schema['system'].keys():
                    raise ValueError(
                        f'unknown system: {system} '
                        f'while parsing schema os:{value}')
        if key in ('nccl', 'cutensor', 'cusparselt', 'cudnn'):
            for value, value_schema in key_schema.items():
                for cuda, _ in value_schema.get('cuda', {}).items():
                    if cuda not in schema['cuda'].keys():
                        raise ValueError(
                            f'unknown CUDA version: {cuda} '
                            f'while parsing schema {key}:{value}')
        elif key in ('numpy', 'scipy'):
            for value, value_schema in key_schema.items():
                for python in value_schema.get('python', []):
                    if python not in schema['python'].keys():
                        raise ValueError(
                            f'unknown Python version: {python} '
                            f'while parsing schema {key}:{value}')
                for numpy in value_schema.get('numpy', []):
                    if numpy not in schema['numpy'].keys():
                        raise ValueError(
                            f'unknown NumPy version: {numpy} '
                            f'while parsing schema {key}:{value}')


def validate_matrixes(schema: SchemaType, matrixes: List[Matrix]) -> None:
    """Validate matrixes for consistency and raise ValueError if inconsistencies are found.
    Parameters:
        - schema (SchemaType): A dictionary of dictionaries containing possible values for each key.
        - matrixes (List[Matrix]): A list of Matrix objects to be validated.
    Returns:
        - None: This function does not return anything, it only raises ValueError if inconsistencies are found.
    Processing Logic:
        - Check for overall consistency.
        - Check for consistency for each matrix.
        - Raise ValueError if inconsistencies are found."""

    # Validate overall consistency
    project_seen = set()
    system_target_seen = set()
    for matrix in matrixes:
        if not hasattr(matrix, 'project'):
            raise ValueError(f'matrix must have a project: {matrix}')

        if matrix.project in project_seen:
            raise ValueError(f'{matrix.project}: duplicate project name')
        project_seen.add(matrix.project)

        if not hasattr(matrix, 'target'):
            raise ValueError(f'{matrix.project}: target is missing')

        if (matrix.system, matrix.target) in system_target_seen:
            raise ValueError(
                f'{matrix.project}: duplicate system/target combination: '
                f'{matrix.system}/{matrix.target}')
        system_target_seen.add((matrix.system, matrix.target))

        if not hasattr(matrix, 'tags'):
            raise ValueError(f'{matrix.project}: tags is missing')

    # Validate consistency for each matrix
    for matrix in matrixes:
        if matrix._extern:
            continue

        if matrix.cuda is None and matrix.rocm is None:
            raise ValueError(
                f'{matrix.project}: Either cuda nor rocm must be non-null')

        if matrix.cuda is not None and matrix.rocm is not None:
            raise ValueError(
                f'{matrix.project}: cuda and rocm are mutually exclusive')

        for key, key_schema in schema.items():
            possible_values = list(key_schema.keys())
            if not hasattr(matrix, key):
                raise ValueError(f'{matrix.project}: {key} is missing')
            value = getattr(matrix, key)
            if value not in possible_values:
                raise ValueError(
                    f'{matrix.project}: {key} must be one of '
                    f'{possible_values} but got {value}')

            if key in ('nccl', 'cutensor', 'cusparselt', 'cudnn'):
                supports = schema[key][value].get('cuda', None)
                if supports is not None and matrix.cuda not in supports:
                    raise ValueError(
                        f'{matrix.project}: CUDA {matrix.cuda} '
                        f'not supported by {key} {value}')
            elif key in ('numpy', 'scipy'):
                supports = schema[key][value].get('python', None)
                if supports is not None and matrix.python not in supports:
                    raise ValueError(
                        f'{matrix.project}: Python {matrix.python} '
                        f'not supported by {key} {value}')
                supports = schema[key][value].get('numpy', None)
                if supports is not None and matrix.numpy not in supports:
                    raise ValueError(
                        f'{matrix.project}: NumPy {matrix.numpy} '
                        f'not supported by {key} {value}')


def expand_inherited_matrixes(matrixes: List[Matrix]) -> None:
    """Expands inherited matrixes with parent's values.
    Parameters:
        - matrixes (List[Matrix]): List of Matrix objects to be expanded.
    Returns:
        - None: This function does not return anything.
    Processing Logic:
        - Create a dictionary mapping project names to Matrix objects.
        - Loop through each Matrix object in the list.
        - If the Matrix object does not inherit from another Matrix, skip it.
        - Otherwise, find the parent Matrix object using the dictionary.
        - Make a copy of the parent Matrix object.
        - Update the copy with the values from the current Matrix object.
        - Update the current Matrix object with the values from the copy."""

    prj2mat = {m.project: m for m in matrixes}
    for matrix in matrixes:
        if matrix._inherits is None:
            continue
        parent = prj2mat[matrix._inherits]
        assert parent._inherits is None, 'no nested inheritance'
        # Fill values missing in the matrix with parent's values
        inherited = parent.copy()
        inherited.update(matrix)
        matrix.update(inherited)


def log(msg: str, visible: bool = True) -> None:
    """Prints a message if visible is True.
    Parameters:
        - msg (str): The message to be printed.
        - visible (bool): Determines if the message should be printed or not. Defaults to True.
    Returns:
        - None: Does not return anything.
    Processing Logic:
        - Prints message if visible is True."""

    if visible:
        print(msg)


def parse_args(argv: List[str]) -> Any:
    """Function parses arguments from command line and returns the parsed arguments.
    Parameters:
        - argv (List[str]): List of command line arguments.
    Returns:
        - Any: Parsed arguments.
    Processing Logic:
        - Parses arguments from command line.
        - Sets default values for optional arguments.
        - Returns parsed arguments.
        - Arguments are parsed using argparse library."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--schema', type=str, default=None)
    parser.add_argument('-m', '--matrix', type=str, default=None)
    parser.add_argument('-d', '--directory', type=str)
    parser.add_argument('-D', '--dry-run', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    return parser.parse_args()


def main(argv: List[str]) -> int:
    """Generates test assets, coverage matrix, and tags based on project matrixes.
    Parameters:
        - argv (List[str]): List of command line arguments.
    Returns:
        - int: Return code indicating success or failure.
    Processing Logic:
        - Parse command line arguments.
        - Load schema and project matrixes.
        - Generate test assets for each project matrix.
        - Generate coverage matrix.
        - Generate tags.
        - Write output files to specified directory or current directory.
        - Return success or failure code.
    Example:
        >>> main(['--schema', 'schema.yaml', '--matrix', 'matrix.yaml'])
        0"""

    options = parse_args(argv)

    basedir = os.path.abspath(os.path.dirname(argv[0]))
    if options.schema is None:
        options.schema = os.path.join(basedir, 'schema.yaml')
    if options.matrix is None:
        options.matrix = os.path.join(basedir, 'matrix.yaml')

    log(f'Loading schema: {options.schema}', options.verbose)
    with open(options.schema) as f:
        schema = yaml.load(f, Loader=yaml.SafeLoader)
    validate_schema(schema)

    log(f'Loading project matrixes: {options.matrix}', options.verbose)
    matrixes = []
    with open(options.matrix) as f:
        for matrix_record in yaml.load(f, Loader=yaml.SafeLoader):
            matrixes.append(Matrix(matrix_record))
    expand_inherited_matrixes(matrixes)
    validate_matrixes(schema, matrixes)

    output = {}

    # Generate test assets
    for matrix in matrixes:
        if matrix._extern:
            log(
                f'Skipping unmanaged project matrix: {matrix.project} '
                f'(system: {matrix.system}, target: {matrix.target})',
                options.verbose)
            continue
        log(
            f'Processing project matrix: {matrix.project} '
            f'(system: {matrix.system}, target: {matrix.target})',
            options.verbose)
        if matrix.system == 'linux':
            gen = LinuxGenerator(schema, matrix)
            output[f'linux/tests/{matrix.target}.Dockerfile'] = \
                gen.generate_dockerfile()
            output[f'linux/tests/{matrix.target}.sh'] = \
                gen.generate_script()
        elif matrix.system == 'windows':
            raise ValueError('Windows is not supported yet')
        else:
            raise AssertionError

    # Generate coverage matrix
    covgen = CoverageGenerator(schema, [m for m in matrixes if not m._extern])
    covout, warns = covgen.generate_rst()
    output['coverage.rst'] = covout
    if len(warns) != 0:
        log('----------------------------------------')
        log('Test coverage warnings:')
        for w in warns:
            log(f'* {w}')
        log('----------------------------------------')

    # Generate tags
    taggen = TagGenerator(matrixes)
    output['config.tags.json'] = taggen.generate()

    # Write output files.
    out_basedir = options.directory if options.directory else basedir
    retval = 0
    for filename, content in output.items():
        filepath = f'{out_basedir}/{filename}'
        if options.dry_run:
            if os.path.exists(filepath):
                with open(filepath) as f:
                    if f.read() != content:
                        log(f'{filepath} needs to be regenerated')
                        retval = 1
            else:
                log(f'{filepath} needs to be generated')
                retval = 1
        else:
            log(f'Writing {filepath}', options.verbose)
            with open(filepath, 'w') as f:
                f.write(content)
    return retval


if __name__ == '__main__':
    sys.exit(main(sys.argv))
