# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**smallvlm** is an early-stage project for training small Vision Language Models (VLMs) on chest X-ray medical imaging data. The current code is a data ingestion starting point; the broader goal is to build a training pipeline on top of it.

## Running the Code

```bash
# Download the Indiana University chest X-ray dataset from Kaggle
python download.py
```

Requires `kagglehub` and valid Kaggle credentials configured locally (`~/.kaggle/kaggle.json`).

## Dataset

- Source: Kaggle dataset `raddar/chest-xrays-indiana-university`
- Downloaded via `kagglehub.dataset_download()`, which caches to `~/.cache/kagglehub/`
- The downloaded path is printed to stdout for use in downstream steps
