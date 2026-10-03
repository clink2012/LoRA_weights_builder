import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("questionnaire_server", Path(__file__).with_name("server.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QuestionnairePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.server = module.QuestionnaireServer(0, self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def request(self, value=None, route="/api/answers", **headers):
        data = None if value is None else json.dumps(value).encode()
        request = Request(self.url + route, data=data, headers={"Content-Type": "application/json", **headers})
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    def payload(self, choice="violet", submitted=False):
        return {"schema_version": 1, "answers": {"direction": choice}, "notes": "Personal preference", "submitted": submitted}

    def test_initial_state_is_unanswered(self):
        value = self.request()
        self.assertEqual(value["answers"], {})
        self.assertFalse(value["submitted"])
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_draft_and_completion_are_durable_and_history_is_retained(self):
        self.request(self.payload())
        self.assertEqual(self.request()["answers"], {"direction": "violet"})
        self.request(self.payload(submitted=True))
        self.request(self.payload("emerald", submitted=True))
        snapshots = list((self.root / "submissions").glob("completed-*.json"))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual({json.loads(p.read_text())["answers"]["direction"] for p in snapshots}, {"violet", "emerald"})
        self.assertEqual(json.loads((self.root / "answers.json").read_text())["answers"]["direction"], "emerald")

    def test_invalid_answers_do_not_replace_saved_draft(self):
        self.request(self.payload())
        bad = self.payload(); bad["answers"] = {"direction": {"unexpected": "object"}}
        with self.assertRaises(HTTPError) as error:
            self.request(bad)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.request()["answers"]["direction"], "violet")

    def test_nonlocal_origin_cannot_write_answers(self):
        with self.assertRaises(HTTPError) as error:
            self.request(self.payload(), Origin="https://example.com")
        self.assertEqual(error.exception.code, 403)
        self.assertFalse((self.root / "answers.json").exists())

    def test_answers_are_not_exposed_as_arbitrary_files(self):
        self.request(self.payload())
        for route in ["/../../.git/config", "/answers.json", "/references/../../answers.json"]:
            with self.assertRaises(HTTPError) as error:
                self.request(route=route)
            self.assertEqual(error.exception.code, 404)

    def test_corrupt_saved_data_is_reported_without_overwrite(self):
        (self.root / "answers.json").write_text("invalid", encoding="utf-8")
        with self.assertRaises(HTTPError) as error:
            self.request()
        self.assertEqual(error.exception.code, 500)
        self.assertEqual((self.root / "answers.json").read_text(), "invalid")

    def test_valid_json_with_invalid_schema_is_preserved_and_reported(self):
        content = '{"schema_version":1,"answers":[]}'
        (self.root / "answers.json").write_text(content, encoding="utf-8")
        with self.assertRaises(HTTPError) as error:
            self.request()
        self.assertEqual(error.exception.code, 500)
        self.assertEqual((self.root / "answers.json").read_text(), content)


if __name__ == "__main__":
    unittest.main()
