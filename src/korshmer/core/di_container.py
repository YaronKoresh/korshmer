import threading


class CircularDependencyError(RuntimeError):
    pass


class DIContainer:
    def __init__(self):
        self._services = {}
        self._singletons = {}
        self._resolving = []
        self._lock = threading.RLock()

    def register(self, name, factory, singleton=True):
        with self._lock:
            self._services[name] = (factory, singleton)

    def register_instance(self, name, instance):
        with self._lock:
            self._singletons[name] = instance
            self._services[name] = (None, True)

    def get(self, name):
        with self._lock:
            if name in self._singletons:
                return self._singletons[name]

            if name not in self._services:
                raise KeyError(f"Service '{name}' not registered")

            factory, is_singleton = self._services[name]
            if factory is None:
                raise KeyError(f"Service '{name}' not registered")

            if name in self._resolving:
                chain = " -> ".join([*self._resolving, name])
                raise CircularDependencyError(
                    f"Circular dependency detected: {chain}"
                )

            self._resolving.append(name)
            try:
                instance = factory()
            finally:
                self._resolving.pop()

            if is_singleton:
                self._singletons[name] = instance

            return instance

    def has(self, name):
        with self._lock:
            return name in self._services or name in self._singletons

    def clear(self):
        with self._lock:
            self._services.clear()
            self._singletons.clear()
