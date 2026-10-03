# ml_platform_error_assistant

[![Github Actions Status](/workflows/Build/badge.svg)](/actions/workflows/build.yml)

JupyterLab error assistant frontend and server extension

This extension is composed of a Python package named `ml_platform_error_assistant`
for the server extension and a NPM package named `@ml-platform/error-assistant`
for the frontend extension.

## Requirements

- JupyterLab >= 4.0.0

## Install

To install the extension, execute:

```bash
pip install ml_platform_error_assistant
```

## Uninstall

To remove the extension, execute:

```bash
pip uninstall ml_platform_error_assistant
```

## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

If the server extension is installed and enabled, but you are not seeing
the frontend extension, check the frontend extension is installed:

```bash
jupyter labextension list
```

## Contributing

If you would like to contribute to this extension, please refer to the [Contributing Guide](CONTRIBUTING.md).
