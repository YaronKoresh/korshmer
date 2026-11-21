import hashlib
import hmac
import importlib.util
import json
import logging
import os
import sys

from korshmer.plugins.base import (
    AudioEffectPlugin,
    ExporterPlugin,
    MidiProcessorPlugin,
)
from korshmer.utils.audit_log import OUTCOME_DENIED, OUTCOME_FAILED, audit

_logger = logging.getLogger(__name__)

USER_PLUGINS_ENV_VAR = "KORSHMER_ENABLE_USER_PLUGINS"
APPROVED_PLUGINS_FILENAME = "approved_plugins.json"
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_PLUGIN_BASES = (MidiProcessorPlugin, AudioEffectPlugin, ExporterPlugin)


def user_plugins_enabled(environ=None):
    env = os.environ if environ is None else environ
    value = (env.get(USER_PLUGINS_ENV_VAR) or "").strip().lower()
    return value in _ENABLED_VALUES


def approved_plugins_path(plugin_dir):
    parent = os.path.dirname(os.path.abspath(plugin_dir))
    return os.path.join(parent, APPROVED_PLUGINS_FILENAME)


def read_approved_plugins(plugin_dir):
    try:
        with open(approved_plugins_path(plugin_dir), encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        _logger.warning(f"Ignoring unreadable plugin approval list: {e}")
        return {}
    if not isinstance(data, dict):
        _logger.warning("Ignoring plugin approval list: not a JSON object")
        return {}
    return {
        str(name): digest.strip().lower()
        for name, digest in data.items()
        if isinstance(digest, str)
    }


class PluginRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {}
            cls._instance._loaded = False
        return cls._instance

    def register(self, plugin_cls):
        if (
            not hasattr(plugin_cls, "plugin_type")
            or plugin_cls.plugin_type is None
        ):
            _logger.warning(f"Skipping {plugin_cls.__name__}: no plugin_type")
            return False

        key = f"{plugin_cls.plugin_type.name}:{plugin_cls.name}"
        if key in self._plugins:
            _logger.warning(f"Plugin already registered: {key}")
            return False

        self._plugins[key] = plugin_cls
        _logger.info(f"Registered plugin: {key}")
        return True

    def unregister(self, plugin_cls):
        key = f"{plugin_cls.plugin_type.name}:{plugin_cls.name}"
        return self._plugins.pop(key, None) is not None

    def get_plugins(self, plugin_type=None, category=None):
        results = []
        for key, cls in self._plugins.items():
            if plugin_type and not key.startswith(plugin_type.name):
                continue
            if category and cls.category != category:
                continue
            results.append(cls)
        return results

    def get_plugin(self, name, plugin_type=None):
        if plugin_type:
            key = f"{plugin_type.name}:{name}"
            return self._plugins.get(key)

        for _key, cls in self._plugins.items():
            if cls.name == name:
                return cls
        return None

    def get_categories(self, plugin_type=None):
        cats = set()
        for cls in self.get_plugins(plugin_type):
            cats.add(cls.category)
        return sorted(cats)

    def discover_entry_points(self, group="korshmer.plugins"):
        from importlib.metadata import entry_points

        eps = entry_points(group=group)

        count = 0
        for ep in eps:
            try:
                plugin_cls = ep.load()
                audit("plugin.load", source="entry_point", name=ep.name)
                if self.register(plugin_cls):
                    count += 1
            except Exception as e:
                _logger.error(f"Failed to load plugin {ep.name}: {e}")
                audit(
                    "plugin.load",
                    OUTCOME_FAILED,
                    source="entry_point",
                    name=ep.name,
                    error=type(e).__name__,
                )
        return count

    def _run_plugin_source(self, module_name, fpath, source):
        spec = importlib.util.spec_from_file_location(module_name, fpath)
        if spec is None:
            raise ImportError(f"Cannot load {fpath}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        try:
            exec(compile(source, fpath, "exec"), mod.__dict__)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return mod

    def discover_directory(self, path, approved=None):
        if not os.path.isdir(path):
            return 0

        count = 0
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            module_name = f"korshmer_plugin_{fname[:-3]}"
            fpath = os.path.join(path, fname)
            try:
                with open(fpath, "rb") as f:
                    source = f.read()
                digest = hashlib.sha256(source).hexdigest()
                if approved is not None and not hmac.compare_digest(
                    approved.get(fname, ""), digest
                ):
                    _logger.warning(
                        f"Skipping plugin {fname}: not approved "
                        f"(sha256 {digest}). Add it to "
                        f"{approved_plugins_path(path)} or set "
                        f"{USER_PLUGINS_ENV_VAR}=1 to load it."
                    )
                    audit(
                        "plugin.load",
                        OUTCOME_DENIED,
                        source="directory",
                        path=fpath,
                        sha256=digest,
                    )
                    continue
                mod = self._run_plugin_source(module_name, fpath, source)
                audit("plugin.load", source="directory", path=fpath)
                for attr in list(vars(mod).values()):
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, _PLUGIN_BASES)
                        and attr not in _PLUGIN_BASES
                        and self.register(attr)
                    ):
                        count += 1
            except Exception as e:
                _logger.error(f"Failed to load plugin from {fname}: {e}")
                audit(
                    "plugin.load",
                    OUTCOME_FAILED,
                    source="directory",
                    path=fpath,
                    error=type(e).__name__,
                )
        return count

    def load_all(self, plugin_dir=None):
        if self._loaded:
            return
        self._loaded = True

        from korshmer.plugins.builtin import register_builtins

        nb = register_builtins(self)
        _logger.info(f"Registered {nb} built-in plugins")

        n = self.discover_entry_points()
        _logger.info(f"Discovered {n} plugins from entry_points")

        if plugin_dir:
            approved = (
                None
                if user_plugins_enabled()
                else read_approved_plugins(plugin_dir)
            )
            n2 = self.discover_directory(plugin_dir, approved=approved)
            _logger.info(f"Discovered {n2} plugins from {plugin_dir}")

    def get_by_protocol(self, protocol_cls):
        return [
            cls
            for cls in self._plugins.values()
            if isinstance(cls, type) and issubclass(cls, protocol_cls)
        ]

    def get_metadata(self, name, plugin_type=None):
        cls = self.get_plugin(name, plugin_type)
        if cls is None:
            return None
        from korshmer.plugins.base import PluginMetadata

        return PluginMetadata(
            name=cls.name,
            version=getattr(cls, "version", "1.0.0"),
            author=getattr(cls, "author", ""),
            category=cls.category,
            plugin_type=cls.plugin_type,
            params=cls.params(),
            description=getattr(cls, "description", ""),
        )

    def clear(self):
        self._plugins.clear()
        self._loaded = False

    @classmethod
    def reset(cls):
        if cls._instance:
            cls._instance.clear()
        cls._instance = None
