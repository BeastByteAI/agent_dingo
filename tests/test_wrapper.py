import unittest
from unittest.mock import MagicMock
from agent_dingo.wrapper import DingoWrapper


class TestDingoWrapper(unittest.TestCase):

    def setUp(self):
        self.agent_mock = MagicMock()
        self.wrapper = DingoWrapper(self.agent_mock)
        self.client = self.wrapper.create_app().test_client()

    def test_chat_completion(self):
        messages = [{"role": "user", "content": "Hello"}]
        self.agent_mock.chat.return_value = ("Response content", [])
        
        response = self.wrapper.chat_completion("gpt-3.5-turbo-0613", messages)

        self.assertEqual(response["object"], "chat.completion")
        self.assertIn("usage", response)
        self.assertEqual(response["choices"][0]["message"]["content"], "Response content")

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "The server is running.")

    def test_chat_completion_endpoint(self):
        self.agent_mock.chat.return_value = ("Mocked response", [])
        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello, World!"}]
        }
        response = self.client.post('/chat/completions', json=data)
        json_response = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json_response["choices"][0]["message"]["content"], "Mocked response")