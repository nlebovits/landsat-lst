"""Vulture whitelist for false positives.

Add entries here for code that vulture incorrectly flags as unused.
Format: module.attribute or just attribute
"""

# Pydantic computed fields are used by the framework
_.name  # noqa: F821
_.bbox  # noqa: F821
_.datetime_range  # noqa: F821
_.output_filename  # noqa: F821
_.is_daytime  # noqa: F821

# Click CLI commands are discovered dynamically
_.process  # noqa: F821
_.list_tiles  # noqa: F821
_.tile_info  # noqa: F821

# pytest fixtures
_.tiny_bbox  # noqa: F821
_.pergamino_bbox  # noqa: F821
_.sample_tile  # noqa: F821
_.sample_job  # noqa: F821
_.mock_qa_pixel  # noqa: F821
_.mock_lwir_band  # noqa: F821
_.fixtures_dir  # noqa: F821
_.data_cache_dir  # noqa: F821
