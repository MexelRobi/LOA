import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("mpkgexec")


class MpkgExecTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.home = self.root / "home"
        self.plist = self.home / "Library/LaunchAgents/com.loa.web.plist"
        self.destination = self.root / "loa"
        self.log = self.root / "commands"
        self.command("curl", """\
echo "curl $*" >> "$TEST_LOG"
case "$*" in
  *-w*) printf 200 ;;
  *) while [ "$1" != "-o" ]; do shift; done; printf binary > "$2" ;;
esac
""")
        self.command("launchctl", """\
echo "launchctl $*" >> "$TEST_LOG"
if [ "$1" = asuser ]; then shift 2; exec "$@"; fi
if [ "$1" = print ]; then
  case "$2" in
    */com.loa.web) [ "$TEST_LOADED" = 1 ]; exit $? ;;
    *) [ "$TEST_DOMAIN" = 1 ]; exit $? ;;
  esac
fi
if [ "$1" = bootout ] && [ "$TEST_BOOTOUT_FAIL" = 1 ]; then exit 1; fi
if [ "$1" = bootstrap ] && [ "$TEST_BOOTSTRAP_FAIL" = 1 ]; then exit 1; fi
""")
        self.command("sudo", """\
echo "sudo $*" >> "$TEST_LOG"
if [ "$1" = -u ]; then shift 2; fi
exec "$@"
""")
        self.command("install", """\
echo "install $*" >> "$TEST_LOG"
if [ "$TEST_INSTALL_FAIL" = 1 ]; then exit 1; fi
/usr/bin/install "$@"
""")
        self.command("xattr", "exit 0\n")
        self.command("id", """\
case "$*" in
  -un) printf '%s\\n' "$TEST_USER" ;;
  -u*) printf '%s\\n' "$TEST_UID" ;;
esac
""")
        self.command("dscl", 'printf "NFSHomeDirectory: %s\\n" "$TEST_HOME"\n')
        self.script = self.root / "mpkgexec"
        self.script.write_text(
            SCRIPT.read_text()
            .replace('DEST="/usr/local/bin/loa"', f'DEST="{self.destination}"')
            .replace('LAUNCHCTL="/bin/launchctl"', f'LAUNCHCTL="{self.bin / "launchctl"}"')
        )
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.home),
            "TEST_HOME": str(self.home),
            "TEST_LOG": str(self.log),
            "TEST_USER": "testuser",
            "TEST_UID": "501",
            "TEST_LOADED": "0",
            "TEST_DOMAIN": "1",
            "TEST_INSTALL_FAIL": "0",
            "TEST_BOOTSTRAP_FAIL": "0",
            "TEST_BOOTOUT_FAIL": "0",
        }

    def command(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)

    def install_plist(self):
        self.plist.parent.mkdir(parents=True)
        self.plist.write_text("original configuration")

    def run_installer(self):
        return subprocess.run(
            ["/bin/bash", str(self.script)], env=self.env, text=True, capture_output=True
        )

    def test_no_agent_only_installs_binary(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.log.read_text()
        self.assertNotIn("launchctl bootout", log)
        self.assertNotIn("launchctl bootstrap", log)
        self.assertEqual(self.destination.read_text(), "binary")

    def test_loaded_agent_stops_then_restarts_with_original_plist(self):
        self.install_plist()
        self.env["TEST_LOADED"] = "1"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.log.read_text()
        self.assertLess(log.index("launchctl bootout"), log.index("install -m"))
        self.assertLess(log.index("install -m"), log.index("launchctl bootstrap"))
        self.assertIn(f"launchctl bootstrap gui/501 {self.plist}", log)
        self.assertIn("launchctl kickstart -k gui/501/com.loa.web", log)
        self.assertEqual(self.plist.read_text(), "original configuration")

    def test_installed_but_unloaded_agent_is_started(self):
        self.install_plist()
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.log.read_text()
        self.assertNotIn("launchctl bootout", log)
        self.assertIn(f"sudo {self.bin / 'launchctl'} bootstrap gui/501", log)
        self.assertIn("launchctl bootstrap", log)

    def test_failed_install_restores_previously_loaded_agent(self):
        self.install_plist()
        self.env.update(TEST_LOADED="1", TEST_INSTALL_FAIL="1")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("launchctl bootstrap", self.log.read_text())

    def test_loaded_agent_without_plist_is_not_stopped(self):
        self.env["TEST_LOADED"] = "1"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("plist is missing", result.stderr)
        self.assertNotIn("launchctl bootout", self.log.read_text())
        self.assertFalse(self.destination.exists())

    def test_launchctl_failure_stops_install(self):
        self.install_plist()
        self.env.update(TEST_LOADED="1", TEST_BOOTOUT_FAIL="1")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not stop", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_unavailable_gui_domain_is_reported(self):
        self.install_plist()
        self.env["TEST_DOMAIN"] = "0"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot access the LaunchAgent domain", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_bootstrap_failure_is_reported(self):
        self.install_plist()
        self.env.update(TEST_LOADED="1", TEST_BOOTSTRAP_FAIL="1")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not register", result.stderr)
        self.assertIn("Could not restore", result.stderr)

    def test_root_uses_sudo_users_gui_domain(self):
        self.install_plist()
        self.env.update(TEST_UID="0", TEST_USER="root", SUDO_USER="testuser", TEST_LOADED="1")
        self.command("id", """\
case "$*" in
  -un) echo root ;;
  -u) echo 0 ;;
  "-u testuser") echo 501 ;;
esac
""")
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.log.read_text()
        self.assertIn("launchctl asuser 501 sudo -u testuser", log)
        self.assertIn(f"launchctl bootstrap gui/501 {self.plist}", log)


if __name__ == "__main__":
    unittest.main()
