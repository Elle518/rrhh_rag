# RRHH RAG

## Prerrequisites

Check if you have Python 3.11 installed:

```bash
> python3.11 --version
> which python3.11
```

Create a virtual environment:

```bash
> python3.11 -m venv .venv
```

Activate the virtual environment:

```bash
> source .venv/bin/activate
```

Check Python version:

```bash
> python --version
> which python
```

Install dependencies:

```bash
> pip install -r requirements.txt
```

For VSCode notebooks extension install:

```bash
> pip install ipykernel ipywidgets jupyter
```

Install Docling for MacOS macOS x86_64:

```bash
> pip install docling "numpy<2.0.0"
```

Source: <https://docling-project.github.io/docling/faq/#is-macos-x86_64-supported>

For other platforms:

```bash
> pip install docling
```

## Using Git Hooks

This repository uses `pre-commit` hooks to format code before each commit. You need to install and configure these hooks before making changes to the repository.

Once you've created/cloned the repository, run the following command to configure the `pre-commit` hooks (you only need to do this once):

```bash
> pip install pre-commit
```

To validate, before pushing changes to the repository, run:

```bash
> pre-commit run --all-files
```

This will run all configured hooks over the repository tracked files. If you want to do it only on a specific file, you can use:

```bash
> pre-commit run --files path/to/file
```

If you want to do it on the modified files in the staging area, you can use:

```bash
> pre-commit run
```

In this case, you don't need to specify the files.

If it passes the checks, you can commit and push as usual. If not, it will correct any errors it finds, and you'll have to push the changes back to the staging area.

If you want to skip the pre-commit hooks on a specific commit, you can use the `--no-verify` option when committing:

```bash
git commit --no-verify -m "my_commit"
```

## Environment variables

Create a `.env` file and populate the following environment variables found in `.env_template` with your OpenAI and Qdrant credentials:

## Data folder

The data folder contains the following subfolders:

* `raw`. Files in regular extensions, like PDF.
* `interm`. Converted files to markdown and JSON.
* `processed`. Chunks of several chunking experiments in JSON format.

To create this folder structure, you can run the following command:

```bash
> python src/utils.py
```
