import os
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from socratic_agent import SocraticAgentManager


class SocraticAgentTests(unittest.TestCase):
    def test_default_model_is_gemma(self):
        agent = SocraticAgentManager()
        self.assertEqual(agent.model_name, "google/gemma-3-1b")

    def test_custom_model_parameter(self):
        agent = SocraticAgentManager(model_name="custom-model-id")
        self.assertEqual(agent.model_name, "custom-model-id")

    def test_env_var_model_override(self):
        os.environ["LM_STUDIO_MODEL"] = "llama-3.2-1b-instruct"
        try:
            agent = SocraticAgentManager()
            self.assertEqual(agent.model_name, "llama-3.2-1b-instruct")
        finally:
            del os.environ["LM_STUDIO_MODEL"]

    def test_state_machine_transitions_with_offline_fallback(self):
        agent = SocraticAgentManager(base_url="http://127.0.0.1:9999/v1")
        session = agent.init_session("test-session-1", child_age=10, threat_type="violence")
        self.assertEqual(session["current_phase"], "Acknowledge")

        # Turn 1: Acknowledge -> Reason
        updated_session, result = agent.execute_turn("test-session-1", "I was just playing a game")
        self.assertEqual(result["state_info"]["previous_phase"], "Acknowledge")
        self.assertEqual(result["state_info"]["next_phase"], "Reason")

        # Turn 2: Reason -> Contract
        updated_session, result = agent.execute_turn("test-session-1", "I understand it's not good")
        self.assertEqual(result["state_info"]["previous_phase"], "Reason")
        self.assertEqual(result["state_info"]["next_phase"], "Contract")

        # Turn 3: Contract agreement
        updated_session, result = agent.execute_turn("test-session-1", "Yes, I agree to close it")
        self.assertEqual(result["agreed_to_boundary"], True)
        self.assertEqual(result["state_info"]["completed"], True)

    def test_turn_alternation_for_jinja_compatibility(self):
        agent = SocraticAgentManager()
        raw_history = [
            {"role": "user", "content": "Alert: threat detected"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "I was just playing a game"},
        ]
        formatted = agent._format_messages_for_llm("SYSTEM PROMPT", raw_history, "violence")
        
        # Must start with system
        self.assertEqual(formatted[0]["role"], "system")
        
        # After system, must start with user
        self.assertEqual(formatted[1]["role"], "user")
        
        # Loop messages after system must strictly alternate: user, assistant, user, assistant...
        for i, msg in enumerate(formatted[1:]):
            if i % 2 == 0:
                self.assertEqual(msg["role"], "user", f"Message at index {i} after system should be user, got {msg['role']}")
            else:
                self.assertEqual(msg["role"], "assistant", f"Message at index {i} after system should be assistant, got {msg['role']}")

    def test_turn_alternation_fixes_leading_assistant_message(self):
        agent = SocraticAgentManager()
        # If history accidentally began with assistant:
        raw_history = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "I am watching a movie"},
        ]
        formatted = agent._format_messages_for_llm("SYSTEM PROMPT", raw_history, "violence")
        
        # Verify first message after system is user, strictly alternating
        for i, msg in enumerate(formatted[1:]):
            expected = "user" if i % 2 == 0 else "assistant"
            self.assertEqual(msg["role"], expected)


if __name__ == "__main__":
    unittest.main()
