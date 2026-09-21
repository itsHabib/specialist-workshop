"""Native lifetime checks use fake local clients and make no model calls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import models

HERE = Path(__file__).resolve().parent
FAKE_CLIENT = r'''
import json, os, stat, subprocess, sys, time
from pathlib import Path
marker = Path(os.environ['BRIDGE_FAKE_MARKER'])
mode = os.environ['BRIDGE_FAKE_MODE']
marker.write_text(json.dumps({'pid': os.getpid(), 'stdin_is_file': stat.S_ISREG(os.fstat(0).st_mode)}))
if mode == 'ignore-stdin':
    while True: time.sleep(1)
prompt = sys.stdin.read()
response = dict(action='hypothesis', hypothesis=str(len(prompt)), prediction='', source='', probes_json='[]')
if mode == 'leave-child':
    child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])
    marker.with_suffix('.child').write_text(str(child.pid))
    time.sleep(.1)
if Path(sys.argv[0]).name == 'codex':
    Path(sys.argv[sys.argv.index('-o')+1]).write_text(json.dumps(response))
    print(json.dumps({'type':'turn.completed','usage':{'input_tokens':13,'output_tokens':8}}))
    raise SystemExit(0)
print(json.dumps({'subtype':'success','is_error':False,'structured_output':response,
                  'modelUsage':{'claude-opus-5':{'inputTokens':13,'outputTokens':8}},
                  'total_cost_usd':0.001}))
'''


class ModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='bridge-model-tests-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.marker = self.root / 'native.json'
        for name in ('claude', 'codex'):
            client = self.root / name
            client.write_text('#!' + sys.executable + '\n' + FAKE_CLIENT)
            client.chmod(0o755)
        self.environment = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                                BRIDGE_FAKE_MODE='success', BRIDGE_FAKE_MARKER=str(self.marker),
                                PYTHONDONTWRITEBYTECODE='1')

    def call(self, provider='opus', mode='success', timeout=5, stop=None, prompt='hello'):
        with patch.dict(os.environ, dict(self.environment, BRIDGE_FAKE_MODE=mode)):
            return models.complete(prompt, provider, self.root / 'call', timeout=timeout, stop=stop)

    def assert_dead(self, pid):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.02)
        self.fail(f'fake native process {pid} is still alive')

    def wait_for(self, path, timeout=5):
        deadline = time.monotonic() + timeout
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(path.exists(), f'{path.name} did not appear')

    def test_opus_success_preserves_response_usage_and_file_stdin(self):
        result = self.call()
        self.assertIsNone(result['error'])
        self.assertEqual(result['response']['hypothesis'], '5')
        self.assertEqual(result['usage']['claude-opus-5']['inputTokens'], 13)
        self.assertEqual(result['estimated_cost_usd'], .001)
        self.assertTrue(json.loads(self.marker.read_text())['stdin_is_file'])
        self.assertEqual(json.loads((self.root / 'call/receipt.json').read_text()), result)

    def test_astra_success_preserves_response_and_usage(self):
        result = self.call(provider='astra')
        self.assertIsNone(result['error'])
        self.assertEqual(result['usage'], {'input_tokens':13,'output_tokens':8})
        self.assertEqual(result['tool_events'], 0)

    def test_timeout_stops_client_that_never_reads_large_stdin(self):
        began = time.monotonic()
        result = self.call(mode='ignore-stdin', timeout=.3, prompt='x'*2_000_000)
        self.assertEqual(result['error'], 'timeout_usage_unknown')
        self.assertLess(time.monotonic() - began, 5)
        self.assert_dead(json.loads(self.marker.read_text())['pid'])

    def test_stop_stops_client_that_never_reads_large_stdin(self):
        stop = self.root / 'STOP'
        def request_stop():
            deadline = time.monotonic() + 4
            while not self.marker.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            stop.touch()
        thread = threading.Thread(target=request_stop)
        thread.start()
        try:
            result = self.call(mode='ignore-stdin', timeout=5, stop=stop, prompt='x'*2_000_000)
        finally:
            thread.join(timeout=5)
        self.assertEqual(result['error'], 'stopped')
        self.assert_dead(json.loads(self.marker.read_text())['pid'])

    def test_preexisting_stop_does_not_launch_native_client(self):
        stop = self.root / 'STOP'; stop.touch()
        result = self.call(stop=stop)
        self.assertEqual(result['error'], 'stopped')
        self.assertIsNone(result['exit_code'])
        self.assertFalse(self.marker.exists())

    def test_driver_sigkill_stops_native_and_retains_receipt(self):
        output = self.root / 'call'
        code = ('import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); '
                'import models; models.complete("x"*2000000, "opus", Path(sys.argv[2]), timeout=15)')
        environment = dict(self.environment, BRIDGE_FAKE_MODE='ignore-stdin')
        driver = subprocess.Popen([sys.executable, '-c', code, str(HERE), str(output)],
                                  env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.wait_for(self.marker)
            native = json.loads(self.marker.read_text())['pid']
            driver.kill(); driver.wait(timeout=2)
            self.wait_for(output / 'receipt.json', timeout=6)
            result = json.loads((output / 'receipt.json').read_text())
            self.assertEqual(result['error'], 'owner_lost_usage_unknown')
            self.assert_dead(native)
        finally:
            if driver.poll() is None:
                driver.kill(); driver.wait(timeout=2)
            if (output / 'process.json').exists() and not (output / 'receipt.json').exists():
                process = json.loads((output / 'process.json').read_text())
                for pid, group in ((process['pid'], True), (process['supervisor_pid'], False)):
                    try:
                        (os.killpg if group else os.kill)(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_normal_client_exit_also_stops_surviving_descendants(self):
        result = self.call(mode='leave-child')
        self.assertIsNone(result['error'])
        self.assert_dead(int(self.marker.with_suffix('.child').read_text()))


if __name__ == '__main__':
    unittest.main()
