from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import share_auth


class ShareAuthTests(unittest.TestCase):
    def test_user_passwords_are_hashed_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"

            share_auth.add_user(store, "alice", "correct horse")

            data = share_auth.load_users(store)
            self.assertIn("alice", data)
            self.assertNotIn("correct horse", store.read_text(encoding="utf-8"))
            self.assertTrue(share_auth.verify_user(store, "alice", "correct horse"))
            self.assertFalse(share_auth.verify_user(store, "alice", "wrong horse"))
            self.assertFalse(share_auth.verify_user(store, "missing", "correct horse"))
            self.assertEqual(share_auth.get_user_role(store, "alice"), "admin")

    def test_user_roles_can_be_managed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"
            share_auth.add_user(store, "admin", "correct horse", role="admin")
            share_auth.add_user(store, "viewer", "correct horse", role="user")

            self.assertEqual(share_auth.get_user_role(store, "admin"), "admin")
            self.assertEqual(share_auth.get_user_role(store, "viewer"), "user")
            self.assertTrue(share_auth.set_user_role(store, "viewer", "admin"))
            self.assertEqual(share_auth.get_user_role(store, "viewer"), "admin")

            records = share_auth.list_user_records(store)
            self.assertEqual([r["username"] for r in records], ["admin", "viewer"])
            self.assertEqual(records[1]["role"], "admin")

    def test_session_token_round_trip_and_revocation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"
            share_auth.add_user(store, "alice", "correct horse")

            token = share_auth.create_session_token("alice", "secret", now=1000)

            self.assertEqual(
                share_auth.verify_session_token(token, "secret", store, now=1001),
                "alice",
            )
            self.assertIsNone(share_auth.verify_session_token(token, "wrong", store, now=1001))
            self.assertIsNone(
                share_auth.verify_session_token(token, "secret", store, now=1000 + share_auth.SESSION_TTL_SECONDS + 1)
            )

            share_auth.remove_user(store, "alice")
            self.assertIsNone(share_auth.verify_session_token(token, "secret", store, now=1001))

    def test_rejects_short_passwords(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                share_auth.add_user(Path(td) / "auth.json", "alice", "short")


if __name__ == "__main__":
    unittest.main()
