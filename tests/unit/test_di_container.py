import threading

import pytest

from korshmer.container import Container
from korshmer.core.di_container import CircularDependencyError, DIContainer


def test_legacy_import_is_same_class():
    assert Container is DIContainer


def test_nested_dependency_does_not_deadlock():
    c = DIContainer()
    c.register("b", lambda: 1)
    c.register("a", lambda: c.get("b"))
    result = []
    t = threading.Thread(target=lambda: result.append(c.get("a")), daemon=True)
    t.start()
    t.join(5)
    assert result == [1]


def test_circular_dependency_raises_clear_error():
    c = DIContainer()
    c.register("a", lambda: c.get("b"))
    c.register("b", lambda: c.get("a"))
    with pytest.raises(CircularDependencyError, match="a -> b -> a"):
        c.get("a")

    c.register("a", lambda: 5)
    assert c.get("a") == 5


def test_unregistered_raises_keyerror_with_name():
    with pytest.raises(KeyError, match="missing"):
        DIContainer().get("missing")


def test_singleton_and_transient():
    c = DIContainer()
    c.register("s", object)
    c.register("t", object, singleton=False)
    assert c.get("s") is c.get("s")
    assert c.get("t") is not c.get("t")


def test_register_instance_has_and_clear():
    c = DIContainer()
    obj = object()
    c.register_instance("x", obj)
    assert c.has("x") and c.get("x") is obj
    c.clear()
    assert not c.has("x")
