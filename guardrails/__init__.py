"""OpenBrain Guard Rails — a standalone V2 service that captures and stores the
approval / permission prompts AI agents raise inside IDEs.

V1 scope (this package): faithfully *record* every approval event in a
structured form — the user's request, the agent's intended action, the verbatim
prompt, the options offered, the user's choice, the execution result, and
provenance metadata. Nothing more: risk scoring, policy enforcement and
auto-approval recommendations are deliberately deferred to later versions and
will build on this clean event log.

It runs as its own loopback daemon (own DB, port, token, dashboard) — fully
decoupled from the memory daemon — so the two can be versioned and operated
independently.
"""

__version__ = "0.1.0"
