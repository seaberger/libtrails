"""Tests for configuration management."""

import pytest
import yaml
from pathlib import Path

from libtrails.config import get_user_config, save_user_config, get_ipad_url, set_ipad_url


class TestUserConfig:
    """Tests for user configuration file handling."""

    def test_get_config_missing_file(self, temp_config_dir, monkeypatch):
        """Test getting config when file doesn't exist."""
        # Ensure the config module uses our temp directory
        from libtrails import config
        monkeypatch.setattr(config, 'USER_CONFIG_DIR', temp_config_dir)
        monkeypatch.setattr(config, 'USER_CONFIG_FILE', temp_config_dir / "config.yaml")

        result = get_user_config()
        assert result == {}

    def test_save_and_load_config(self, temp_config_dir, monkeypatch):
        """Test saving and loading configuration."""
        from libtrails import config
        monkeypatch.setattr(config, 'USER_CONFIG_DIR', temp_config_dir)
        monkeypatch.setattr(config, 'USER_CONFIG_FILE', temp_config_dir / "config.yaml")

        test_config = {
            'ipad': {'default_url': 'http://test:8082'},
            'indexing': {'model': 'gemma3:4b'}
        }

        save_user_config(test_config)

        # Verify file was created
        config_file = temp_config_dir / "config.yaml"
        assert config_file.exists()

        # Verify content
        loaded = get_user_config()
        assert loaded == test_config

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        """Test that save creates the config directory if it doesn't exist."""
        from libtrails import config

        new_config_dir = tmp_path / "new_dir" / ".libtrails"
        monkeypatch.setattr(config, 'USER_CONFIG_DIR', new_config_dir)
        monkeypatch.setattr(config, 'USER_CONFIG_FILE', new_config_dir / "config.yaml")

        save_user_config({'test': 'value'})

        assert new_config_dir.exists()
        assert (new_config_dir / "config.yaml").exists()


class TestIpadUrl:
    """Tests for iPad URL configuration."""

    def test_get_ipad_url_not_set(self, temp_config_dir, monkeypatch):
        """Test getting iPad URL when not configured."""
        from libtrails import config
        monkeypatch.setattr(config, 'USER_CONFIG_DIR', temp_config_dir)
        monkeypatch.setattr(config, 'USER_CONFIG_FILE', temp_config_dir / "config.yaml")

        url = get_ipad_url()
        assert url is None

    def test_set_and_get_ipad_url(self, temp_config_dir, monkeypatch):
        """Test setting and getting iPad URL."""
        from libtrails import config
        monkeypatch.setattr(config, 'USER_CONFIG_DIR', temp_config_dir)
        monkeypatch.setattr(config, 'USER_CONFIG_FILE', temp_config_dir / "config.yaml")

        test_url = "http://192.168.1.100:8082"
        set_ipad_url(test_url)

        result = get_ipad_url()
        assert result == test_url

    def test_set_ipad_url_preserves_other_config(self, temp_config_dir, monkeypatch):
        """Test that setting iPad URL doesn't overwrite other config."""
        from libtrails import config
        monkeypatch.setattr(config, 'USER_CONFIG_DIR', temp_config_dir)
        monkeypatch.setattr(config, 'USER_CONFIG_FILE', temp_config_dir / "config.yaml")

        # Set initial config
        save_user_config({
            'other_setting': 'value',
            'indexing': {'model': 'test-model'}
        })

        # Set iPad URL
        set_ipad_url("http://test:8082")

        # Verify other settings preserved
        loaded = get_user_config()
        assert loaded['other_setting'] == 'value'
        assert loaded['indexing']['model'] == 'test-model'
        assert loaded['ipad']['default_url'] == "http://test:8082"
