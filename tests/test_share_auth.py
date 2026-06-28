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
    def test_resolves_auth_policy_from_config_or_users(self) -> None:
        self.assertFalse(share_auth.resolve_auth_enabled(None, has_users=False))
        self.assertTrue(share_auth.resolve_auth_enabled(None, has_users=True))
        self.assertFalse(share_auth.resolve_auth_enabled("false", has_users=True))
        self.assertFalse(share_auth.resolve_auth_enabled("0", has_users=True))
        self.assertTrue(share_auth.resolve_auth_enabled("true", has_users=False))
        self.assertTrue(share_auth.resolve_auth_enabled("yes", has_users=False))

    def test_auto_funnel_requires_auth_and_truthy_config(self) -> None:
        self.assertFalse(share_auth.resolve_auto_funnel_enabled(None, auth_enabled=True))
        self.assertFalse(share_auth.resolve_auto_funnel_enabled("false", auth_enabled=True))
        self.assertFalse(share_auth.resolve_auto_funnel_enabled("true", auth_enabled=False))
        self.assertFalse(share_auth.resolve_auto_funnel_enabled("true", auth_enabled=True, has_admin=False))
        self.assertTrue(share_auth.resolve_auto_funnel_enabled("true", auth_enabled=True, has_admin=True))
        self.assertTrue(share_auth.resolve_auto_funnel_enabled("1", auth_enabled=True, has_admin=True))

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

            self.assertTrue(share_auth.has_admin(store))
            self.assertEqual(share_auth.get_user_role(store, "admin"), "admin")
            self.assertEqual(share_auth.get_user_role(store, "viewer"), "user")
            self.assertTrue(share_auth.set_user_role(store, "viewer", "admin"))
            self.assertEqual(share_auth.get_user_role(store, "viewer"), "admin")

            records = share_auth.list_user_records(store)
            self.assertEqual([r["username"] for r in records], ["admin", "viewer"])
            self.assertEqual(records[1]["role"], "admin")

    def test_child_role_can_be_managed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"
            share_auth.add_user(store, "admin", "correct horse", role="admin")
            share_auth.add_user(store, "kid", "correct horse", role="child")

            self.assertEqual(share_auth.get_user_role(store, "kid"), "child")
            self.assertFalse(share_auth.is_admin(store, "kid"))

            records = share_auth.list_user_records(store)
            self.assertEqual(
                {r["username"]: r["role"] for r in records},
                {"admin": "admin", "kid": "child"},
            )

            self.assertTrue(share_auth.set_user_role(store, "kid", "user"))
            self.assertEqual(share_auth.get_user_role(store, "kid"), "user")
            self.assertTrue(share_auth.set_user_role(store, "kid", "child"))
            self.assertEqual(share_auth.get_user_role(store, "kid"), "child")

    def test_refuses_to_remove_or_demote_last_admin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"
            share_auth.add_user(store, "admin", "correct horse", role="admin")
            share_auth.add_user(store, "viewer", "correct horse", role="user")

            with self.assertRaisesRegex(ValueError, "last admin"):
                share_auth.set_user_role(store, "admin", "user")
            with self.assertRaisesRegex(ValueError, "last admin"):
                share_auth.remove_user(store, "admin")

            self.assertTrue(share_auth.has_admin(store))
            self.assertEqual(share_auth.get_user_role(store, "admin"), "admin")

    def test_refuses_to_demote_last_admin_to_child(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"
            share_auth.add_user(store, "admin", "correct horse", role="admin")

            with self.assertRaisesRegex(ValueError, "last admin"):
                share_auth.set_user_role(store, "admin", "child")

            self.assertEqual(share_auth.get_user_role(store, "admin"), "admin")

    def test_session_token_round_trip_and_revocation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"
            share_auth.add_user(store, "alice", "correct horse")
            share_auth.add_user(store, "bob", "correct horse", role="admin")

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

    def test_rejects_unsafe_usernames(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "auth.json"
            for username in ("bad name", "../admin", "alice\r\nSet-Cookie:evil=1"):
                with self.subTest(username=username):
                    with self.assertRaises(ValueError):
                        share_auth.add_user(store, username, "correct horse")


if __name__ == "__main__":
    unittest.main()
