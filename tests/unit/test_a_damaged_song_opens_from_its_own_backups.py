from korshmer.project_management.project_manager import ProjectManager


def saved_song(manager, name, tempo):
    manager.start_song(name)
    manager.current_project_data["metadata"]["bpm"] = tempo
    assert manager.save_project()
    return manager.current_project_path


def backed_up(manager):
    manager.last_backup_time = 0
    assert manager.create_backup()


def test_a_damaged_song_opens_from_the_backup_next_to_it(tmp_path):
    manager = ProjectManager(tmp_path)
    damaged = saved_song(manager, "Ballad", 70)
    backed_up(manager)
    saved_song(manager, "Dance", 128)
    backed_up(manager)
    damaged.write_bytes(b"not a song")

    data = manager.load_project_with_fallback(damaged)

    assert data is not None
    assert data["metadata"]["bpm"] == 70
    assert manager.last_load_backup.parent == damaged.parent / "Backups"
    assert manager.current_project_path == damaged


def test_a_damaged_song_opens_from_its_backup_when_no_song_is_open(tmp_path):
    first = ProjectManager(tmp_path)
    damaged = saved_song(first, "Ballad", 70)
    backed_up(first)
    damaged.write_bytes(b"not a song")

    data = ProjectManager(tmp_path).load_project_with_fallback(damaged)

    assert data is not None
    assert data["metadata"]["bpm"] == 70
