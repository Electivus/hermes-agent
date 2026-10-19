"""A temp HERMES_HOME holding third-party plugins, shared by the plugin-host tests of several areas."""

import hermes_yaml as yaml

from hermes_cli import plugins as plugins_mod


def home_with_plugins(tmp_path, monkeypatch, plugins: dict, isolation="host"):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    for name, body in plugins.items():
        plugin_dir = home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": list(plugins), "isolation": isolation}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    return home
