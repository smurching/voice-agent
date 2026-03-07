# Customer Support Voice Agent

A customer support agent built with the OpenAI Agents SDK, deployed on Databricks Apps.

## Overview

This agent helps customers with:
- Order status and tracking
- Returns and refunds
- Product information
- Account management
- Billing and payment questions
- Technical support

## Setup

```bash
uv run quickstart --profile dogfood
```

## Run Locally

```bash
uv run start-app
```

## Deploy

```bash
databricks bundle deploy
databricks bundle run agent_customer_support
```
