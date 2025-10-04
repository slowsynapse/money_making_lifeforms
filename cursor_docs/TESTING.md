# Testing Procedure

This document outlines the procedure for running the integration tests for the Self-Improving Coding Agent.

## Prerequisites

Before running the tests, you must have the following set up:

1.  **Docker**: The tests run inside a Docker container, so you must have Docker installed and running.
2.  **`.env` File**: A `.env` file must be present in the root of the project directory. This file must contain valid API keys for the LLM services the agent uses (e.g., `ANTHROPIC_API_KEY`).
3.  **Docker Image**: You must have the `sica_sandbox` Docker image built. If you have not built it, or if you have made changes to the `Dockerfile`, run the following command from the project root:
    ```bash
    make image
    ```

## Running the Tests

The test suite is run inside the Docker container to ensure a consistent and correct environment.

### Running the Full Test Suite

To run all tests in the `base_agent/tests` directory, use the following command:

```bash
docker run --rm --env-file .env -v "$(pwd)/base_agent":/home/agent/agent_code:rw sica_sandbox pytest agent_code
```

### Running a Specific Test File

To run a specific test file, such as the integration test we created, modify the command to point to that file:

```bash
docker run --rm --env-file .env -v "$(pwd)/base_agent":/home/agent/agent_code:rw sica_sandbox pytest agent_code/tests/test_agent_integration.py
```

### How It Works

-   `docker run --rm`: Runs a new container and automatically removes it when it exits.
-   `--env-file .env`: Loads the environment variables (your API keys) from the `.env` file into the container.
-   `-v "$(pwd)/base_agent":/home/agent/agent_code:rw`: Mounts your local `base_agent` directory into the container at `/home/agent/agent_code`, allowing the test runner to see your code.
-   `sica_sandbox`: The name of the Docker image to use.
-   `pytest agent_code/...`: The command to execute inside the container, which runs the `pytest` test runner on the specified directory or file.
