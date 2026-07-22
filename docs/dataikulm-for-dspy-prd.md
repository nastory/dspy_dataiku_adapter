# DataikuLM for DSPy — Product Requirements Document

**Version:** 1.0  
**Status:** Draft  
**Date:** 2026-07-22

---

## 1. Overview

DataikuLM for DSPy is a Python-based adapter that enables Dataiku users to run DSPy's analysis, teleprompting, optimization, and agent workflows using Dataiku-native LLM Mesh models. The product bridges DSPy's LM abstraction and Dataiku's governed model access so teams can use DSPy's ecosystem without relying on provider-specific integrations that are limited to OpenAI.

The adapter is designed to be as feature-complete as possible for DSPy core capabilities while avoiding changes to either Dataiku LLM Mesh or the DSPy library itself. It aims to provide a simple, consistent interface for data scientists, ML engineers, and AI practitioners building robust prompt programs and optimization pipelines inside Dataiku.

## 2. Problem Statement

DSPy offers powerful tools for prompt programming, teleprompting, optimizers, and agentic workflows, but default integration paths can be constrained by provider-specific assumptions. Dataiku users who depend on LLM Mesh for governance, routing, and enterprise controls need a native way to use DSPy with Mesh-backed models.

Without an adapter, teams face brittle workarounds, reduced feature access, or duplicated integration effort. This project solves that gap by providing a dedicated DataikuLM interface that maps DSPy abstractions to LLM Mesh calls with high parity for core DSPy features.

## 3. Goals & Success Metrics

| Goal | Metric / KPI |
|------|--------------|
| Deliver broad DSPy compatibility on Dataiku LLM Mesh | >= 90% DSPy core feature coverage in validation matrix |
| Enable Dataiku-native DSPy usage without modifying upstream libraries | 100% of integration achieved via adapter layer only |
| Provide a practical interface for robust analysis and optimization workflows | All must-have feature groups validated in acceptance tests |

## 4. Target Users

Primary users include:

- Data scientists building and iterating DSPy programs in Dataiku projects.
- ML engineers operationalizing DSPy-based workflows with enterprise model governance.
- AI practitioners who want DSPy's optimization ecosystem while using Dataiku LLM Mesh as the model access layer.

## 5. Scope

### In Scope

- Build a Python-based adapter, `DataikuLM`, that extends DSPy base classes/interfaces to use Dataiku LLM Mesh models.
- Target compatibility with the latest DSPy release.
- Ensure compatibility with Dataiku DSS 14+.
- Support Python 3.10+.
- Achieve high DSPy core parity with a target of at least 90% feature coverage.
- Include must-have feature groups:
  - Teleprompting and core optimizers
  - Agent workflows (ReAct, multi-step orchestration)
  - Tool/function calling
  - Structured outputs (JSON/schema-constrained)
  - Embeddings/retrieval-related interfaces
- Include nice-to-have features where feasible:
  - Async execution and batching
  - Streaming responses
  - Caching and usage/token accounting

### Out of Scope

- Any changes to or development of Dataiku LLM Mesh itself.
- Any changes to or development of the DSPy library.
- Building custom model providers outside what LLM Mesh supports.

## 6. Features & User Stories

### Feature 1: Core LM Adapter Interface
**User story:** As a Dataiku practitioner, I want to configure DSPy with `DataikuLM` so that DSPy programs can use LLM Mesh models through a simple native interface.

**Acceptance criteria:**
- `DataikuLM` can be instantiated with model/config parameters required by DSPy and LLM Mesh.
- Adapter can be passed to DSPy settings/configuration without upstream code changes.
- Request/response normalization aligns with DSPy expectations for core generation calls.

### Feature 2: Teleprompting and Optimizer Compatibility
**User story:** As a data scientist, I want DSPy teleprompting and optimizer workflows to run unchanged so that I can improve prompt programs inside Dataiku.

**Acceptance criteria:**
- Teleprompting pipelines execute end-to-end with `DataikuLM`.
- Optimizer flows requiring iterative calls and candidate evaluation are supported.
- Feature parity tests for core teleprompting/optimizer scenarios pass.

### Feature 3: Agent Workflow Support
**User story:** As an AI practitioner, I want agentic DSPy workflows to work with `DataikuLM` so that I can run multi-step reasoning and orchestration in Dataiku.

**Acceptance criteria:**
- ReAct-style and multi-step agent flows execute with `DataikuLM`.
- Conversation/message state handling is compatible with DSPy agent expectations.
- Documented behavior exists for any residual unsupported edge cases.

### Feature 4: Tool/Function Calling Support
**User story:** As an ML engineer, I want tool/function calling to be mapped correctly so that DSPy workflows using tools run through LLM Mesh reliably.

**Acceptance criteria:**
- Tool/function call payloads are mapped between DSPy and LLM Mesh schemas.
- Tool call responses are normalized back into DSPy-compatible structures.
- Validation tests cover representative tool-calling patterns.

### Feature 5: Structured Output Support
**User story:** As a DSPy user, I want structured output handling so that schema-constrained and JSON outputs remain reliable.

**Acceptance criteria:**
- Adapter supports JSON/schema-oriented output paths used by DSPy.
- Output parsing and validation behavior is consistent across supported models.
- Failure modes are explicit and documented when model behavior is non-deterministic.

### Feature 6: Embeddings and Retrieval Interface Support
**User story:** As a practitioner building RAG pipelines, I want embeddings/retrieval interfaces to work so that DSPy retrieval components can run via Dataiku-native access.

**Acceptance criteria:**
- Adapter exposes embeddings-related interfaces required by targeted DSPy retrieval flows.
- Retrieval-oriented integration tests pass for supported scenarios.
- Unsupported provider/model cases are surfaced with clear errors.

### Feature 7: Nice-to-Have Runtime Capabilities
**User story:** As a production user, I want async, streaming, caching, and usage accounting where possible so that workflows are efficient and observable.

**Acceptance criteria:**
- Async and batching support is implemented where API support exists.
- Streaming support is available for supported model routes.
- Caching and token/usage metadata are captured when available from LLM Mesh.

## 7. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Performance | No explicit requirement provided; implement efficient request translation and avoid unnecessary overhead in adapter layer. |
| Security | Use Dataiku LLM Mesh-native access patterns and credential handling; do not introduce new secret management mechanisms in adapter code. |
| Scalability | Support repeated optimizer/agent invocation patterns expected in DSPy workflows. |
| Accessibility | Not applicable for backend Python adapter. |
| Compliance | Leverage Dataiku DSS 14+ and LLM Mesh governance controls; no bypass paths outside approved Dataiku model access. |

## 8. Constraints & Dependencies

- Dataiku DSS version must be 14 or newer.
- Python version must be 3.10 or newer.
- Target compatibility is with the most recent DSPy release.
- Adapter must integrate through extension/subclassing and configuration only.
- No code changes are allowed in Dataiku LLM Mesh.
- No code changes are allowed in the DSPy library.
- Final behavior depends on feature support exposed by underlying LLM Mesh model providers.

## 9. Timeline & Milestones

| Milestone | Target Date | Description |
|-----------|-------------|-------------|
| Requirements Finalized | TBD | PRD approved and acceptance criteria baselined. |
| Adapter MVP | TBD | Core `DataikuLM` generation interface and DSPy configuration working. |
| Must-Have Feature Completion | TBD | Teleprompting, optimizers, agents, tool-calling, structured outputs, and retrieval interfaces validated. |
| Parity Validation and Hardening | TBD | Coverage matrix confirms >= 90% core DSPy feature coverage. |
| V1 Release | TBD | Production-ready adapter package and documentation published. |

## 10. Open Questions & Risks

| # | Question / Risk | Owner | Status |
|---|-----------------|-------|--------|
| 1 | Potential response-shape mismatch between DSPy expectations and LLM Mesh outputs may break certain workflows. | TBD | Open |
| 2 | Tool-calling, streaming, and async behavior may have partial parity differences depending on provider/model support in LLM Mesh. | TBD | Open |
| 3 | Structured output reliability may vary by model, impacting deterministic parsing and optimizer consistency. | TBD | Open |
| 4 | Token usage/accounting and caching semantics may not be uniformly available across all routes/providers. | TBD | Open |
| 5 | Unknown edge cases may appear in large-scale agent and optimizer workflows under high iteration counts. | TBD | Open |
