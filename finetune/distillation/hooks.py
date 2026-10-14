"""Scoped activation collection; hooks cannot leak into later inference runs."""

from contextlib import AbstractContextManager


class LayerCapture(AbstractContextManager):
    def __init__(self, model, endpoints, *, detach=False):
        self.layers = model.transformer.layers
        self.endpoints = endpoints
        self.detach = detach
        self.states = {}
        self.handles = []
        if any(index < 1 or index > len(self.layers) for index in endpoints):
            raise ValueError("Layer endpoint outside temporal transformer")

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Activation collector is already active")
        for index in self.endpoints:
            self.handles.append(
                self.layers[index - 1].register_forward_hook(self._hook(index))
            )
        return self

    def _hook(self, index):
        def capture(_module, _inputs, output):
            self.states[index] = output.detach() if self.detach else output

        return capture

    def clear(self):
        self.states.clear()

    def ordered(self):
        if set(self.states) != set(self.endpoints):
            raise RuntimeError("Not all configured layers produced activations")
        return [self.states[index] for index in self.endpoints]

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()
