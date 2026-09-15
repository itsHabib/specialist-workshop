"""Installed, reviewed environments. Package data cannot import executable code."""
from typing import Protocol, Any
from workshop.practice import Practice, diagnostic_rule


class Environment(Protocol):
    done: bool
    def observe(self) -> Any: ...
    def step(self, output: dict) -> None: ...
    def outcome(self) -> dict: ...


class CIEvidence:
    def __init__(self,scenario,directory):self.env=Practice(scenario,directory)
    @property
    def done(self):return self.env.done
    def observe(self):return self.env.observe()
    def step(self,output):self.env.step(output.get('action','invalid-output'))
    def outcome(self):return self.env.outcome()
    def baseline(self):return {'action':diagnostic_rule(self.observe())}


FACTORIES={'ci-evidence-v1':CIEvidence}


def create(name,scenario,directory):
    if name not in FACTORIES:raise ValueError('Environment adapter is not installed')
    return FACTORIES[name](scenario,directory)
