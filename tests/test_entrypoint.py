import runpy


def test_main_module_does_not_run_cli_when_imported():
    namespace = runpy.run_module("topology_pretrain.__main__", run_name="worker_import")
    assert "main" in namespace
