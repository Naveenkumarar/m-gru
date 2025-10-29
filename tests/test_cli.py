import subprocess, sys, tempfile, pathlib

def test_cli_train_and_plot():
    tmp = tempfile.mkdtemp()
    cmd = [sys.executable, "-m", "mgru.cli", "train", "--model", "mgru", "--epochs", "1", "--run", tmp]
    assert subprocess.call(cmd) == 0
    assert subprocess.call([sys.executable, "-m", "mgru.cli", "plot", "--run", tmp]) == 0

