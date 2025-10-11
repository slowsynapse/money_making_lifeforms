# Testing Guide

This document outlines the procedure for running the integration tests for the Self-Improving Coding Agent.

## Prerequisites

Before running the tests, you must have the following set up:

1.  **Docker**: The tests run inside a Docker container, so you must have Docker installed and running.
2.  **`.env` File**: A `.env` file must be present in the root of the project directory. This file must contain valid API keys for any LLM services the agent uses (e.g., `ANTHROPIC_API_KEY`).
3.  **Docker Image**: You must have the `sica_sandbox` Docker image built. If you have not built it, or if you have made changes to the `Dockerfile`, run the following command from the project root:
    ```bash
    make image
    ```

## Running the Tests

The test suite is run inside the Docker container to ensure a consistent and isolated environment. All commands should be run from the project root.

### Running the Full Test Suite

To run all tests in the root `tests/` directory, use the following command:

```bash
docker run --rm --env-file .env -v "$(pwd)":/app sica_sandbox pytest
```

### Running a Specific Test File

To run a specific test file, such as `tests/dsl/test_interpreter.py`, modify the command to point to that file:

```bash
docker run --rm --env-file .env -v "$(pwd)":/app sica_sandbox pytest tests/dsl/test_interpreter.py
```

### How It Works

-   `docker run --rm`: Runs a new container and automatically removes it when it exits.
-   `--env-file .env`: Loads the environment variables (your API keys) from the `.env` file into the container.
-   `-v "$(pwd)":/app`: Mounts your entire local project directory into the container at the `/app` working directory.
-   `sica_sandbox`: The name of the Docker image to use.
-   `pytest ...`: The command to execute inside the container. It runs the `pytest` test runner on the specified directory or file (e.g., the root `tests/` directory or a specific file within it).
