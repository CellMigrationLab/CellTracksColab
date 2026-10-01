
# Contributing to CellTracksColab

Thank you for your interest in contributing to CellTracksColab!

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes and commit with clear messages
5. Push to your fork and submit a pull request

## Where things live

- `notebooks/<Notebook_Name>/`: each notebook, its `requirements.yaml` and its `CHANGELOG.md`. The notebooks run both in Google Colab and in the desktop app, so test changes in both when possible.
- `src/celltracks/`: the `celltracks` Python package used by all notebooks.
- `.tools/` and `.github/workflows/`: packaging and automation from the [LabConstrictor](https://github.com/CellMigrationLab/LabConstrictor) template. These files are kept in sync with the template, so please propose changes to them upstream.

## Guidelines

- Follow existing code style and conventions
- Write clear, descriptive commit messages
- New analysis notebooks should include a test dataset that shows what they do
- Update documentation as needed
- Keep pull requests focused and manageable

## Code Review

All submissions require review. We aim to provide feedback promptly. Please be responsive to review comments.

## Questions?

Feel free to [open an issue](https://github.com/CellMigrationLab/CellTracksColab/issues) for questions or discussions.
