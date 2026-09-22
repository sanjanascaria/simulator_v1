"""Offline regression tests: no credentials or API requests required."""
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / 'reverie/backend_server/persona/prompt_template/gpt_structure.py'
spec = importlib.util.spec_from_file_location('adapter_under_test', path)
api = importlib.util.module_from_spec(spec)
with patch.dict('os.environ', {}, clear=True), patch.dict(sys.modules, {'utils': types.SimpleNamespace(openai_api_key='test-key')}):
    spec.loader.exec_module(api)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.responses.create.return_value = types.SimpleNamespace(
            status='completed', output_text='working\nnext activity')
        self.client_patch = patch.object(api, '_client', self.client)
        self.client_patch.start()
        self.sleep_patch = patch.object(api, 'temp_sleep')
        self.sleep_patch.start()

    def tearDown(self):
        self.client_patch.stop()
        self.sleep_patch.stop()

    def test_completion_contract_and_stop(self):
        self.assertEqual(api.GPT_request('Activity: Isabella is ', {
            'engine': 'text-davinci-003', 'max_tokens': 5, 'stop': ['\n']}), 'working')
        sent = self.client.responses.create.call_args.kwargs
        self.assertEqual(sent['model'], 'gpt-5.6-sol')
        self.assertEqual(sent['reasoning'], {'effort': 'none'})
        self.assertGreaterEqual(sent['max_output_tokens'], 256)
        self.assertIn('missing continuation', sent['instructions'])
        self.assertNotIn('stop', sent)

    def test_all_chat_entry_points_use_sol(self):
        for request in (api.ChatGPT_request, api.GPT4_request, api.ChatGPT_single_request):
            request('hello')
            self.assertEqual(self.client.responses.create.call_args.kwargs['model'], 'gpt-5.6-sol')

    def test_api_errors_escape_retry_wrappers(self):
        self.client.responses.create.side_effect = api.openai.OpenAIError('API unavailable')
        for wrapper in (api.ChatGPT_safe_generate_response, api.GPT4_safe_generate_response):
            with self.assertRaisesRegex(api.openai.OpenAIError, 'API unavailable'):
                wrapper('prompt', 'example', 'instruction')
        with self.assertRaises(api.openai.OpenAIError):
            api.safe_generate_response('prompt', {})
        with self.assertRaises(api.openai.OpenAIError):
            api.ChatGPT_safe_generate_response_OLD('prompt')

    def test_incomplete_and_empty_output_not_used_as_schedule(self):
        self.client.responses.create.return_value = types.SimpleNamespace(
            status='incomplete', output_text='', incomplete_details=types.SimpleNamespace(reason='content_filter'), error=None)
        with self.assertRaises(api.ModelResponseError):
            api.ChatGPT_request('prompt')
        self.client.responses.create.return_value = types.SimpleNamespace(status='completed', output_text='')
        with self.assertRaises(api.ModelResponseError):
            api.ChatGPT_request('prompt')

    def test_token_limit_retries_with_larger_budget(self):
        self.client.responses.create.side_effect = [
            types.SimpleNamespace(status='incomplete', output_text='partial',
                incomplete_details=types.SimpleNamespace(reason='max_output_tokens')),
            types.SimpleNamespace(status='completed', output_text='working')]
        self.assertEqual(api.GPT_request('prompt', {'max_tokens': 50}), 'working')
        budgets = [call.kwargs['max_output_tokens'] for call in self.client.responses.create.call_args_list]
        self.assertEqual(budgets, [1024, 2048])

    def test_stop_before_truncation_is_complete(self):
        self.client.responses.create.return_value = types.SimpleNamespace(
            status='incomplete', output_text='sleeping\n[Next hour] partial',
            incomplete_details=types.SimpleNamespace(reason='max_output_tokens'))
        self.assertEqual(api.GPT_request('prompt', {'stop': ['\n']}), 'sleeping')
        self.assertEqual(self.client.responses.create.call_count, 1)

    def test_truncation_retry_is_bounded(self):
        self.client.responses.create.return_value = types.SimpleNamespace(
            status='incomplete', output_text='partial', error=None,
            incomplete_details=types.SimpleNamespace(reason='max_output_tokens'))
        with self.assertRaises(api.ModelResponseError):
            api.GPT_request('prompt', {})
        self.assertEqual(self.client.responses.create.call_count, 3)

    def test_malformed_output_retries_then_returns_fallback(self):
        cleanup = Mock(side_effect=IndexError('malformed schedule'))
        result = api.safe_generate_response('prompt', {}, repeat=2,
            fail_safe_response=[['working', 60]],
            func_validate=lambda *a, **kw: True, func_clean_up=cleanup)
        self.assertEqual(result, [['working', 60]])
        self.assertEqual(cleanup.call_count, 2)

    def test_embedding_model_preserved(self):
        self.client.embeddings.create.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.1, 0.2])])
        self.assertEqual(api.get_embedding('hello\nworld'), [0.1, 0.2])
        self.client.embeddings.create.assert_called_once_with(
            input=['hello world'], model='text-embedding-ada-002', encoding_format='float')


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        for name, value in [('LLM_PROVIDER', 'ollama'), ('TEXT_MODEL', 'local-chat'),
                            ('EMBEDDING_MODEL', 'local-embed'),
                            ('EMBEDDING_IDENTITY', {'provider': 'ollama', 'model': 'local-embed'}),
                            ('_client', self.client)]:
            patcher = patch.object(api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(api, 'temp_sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def response(self, text='working', reason='stop'):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=text), finish_reason=reason)])

    def test_all_text_entry_points_use_local_chat(self):
        self.client.chat.completions.create.return_value = self.response()
        for request in (api.ChatGPT_request, api.GPT4_request, api.ChatGPT_single_request):
            self.assertEqual(request('hello'), 'working')
        self.assertEqual(api.GPT_request('Activity: ', {'engine': 'old', 'max_tokens': 5}), 'working')
        sent = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent['model'], 'local-chat')
        self.assertIn('missing continuation', sent['messages'][0]['content'])
        self.client.responses.create.assert_not_called()

    def test_local_client_needs_no_openai_key(self):
        with patch.object(api, '_client', None), patch.dict('os.environ', {}, clear=True), patch.object(api.openai, 'OpenAI') as constructor:
            api.get_client()
            constructor.assert_called_once_with(base_url='http://localhost:11434/v1',
                api_key='ollama', timeout=300.0, max_retries=2)

    def test_qwen_thinking_disabled_by_default(self):
        self.client.chat.completions.create.return_value = self.response('8am')
        with patch.object(api, 'TEXT_MODEL', 'qwen3.6:27b'), patch.dict('os.environ', {}, clear=True):
            self.assertEqual(api.GPT_request('Wake up: ', {'max_tokens': 5}), '8am')
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs['reasoning_effort'], 'none')

    def test_thinking_override_and_other_models(self):
        self.client.chat.completions.create.return_value = self.response()
        with patch.dict('os.environ', {}, clear=True):
            api.ChatGPT_request('prompt')
            self.assertNotIn('reasoning_effort', self.client.chat.completions.create.call_args.kwargs)
        with patch.object(api, 'TEXT_MODEL', 'qwen3.6:27b'):
            for effort in ('low', 'default'):
                with patch.dict('os.environ', {'OLLAMA_REASONING_EFFORT': effort}, clear=True):
                    api.ChatGPT_request('prompt')
                sent = self.client.chat.completions.create.call_args.kwargs
                if effort == 'default':
                    self.assertNotIn('reasoning_effort', sent)
                else:
                    self.assertEqual(sent['reasoning_effort'], effort)

    def test_custom_server_url(self):
        with patch.object(api, '_client', None), patch.dict('os.environ', {
                'OLLAMA_BASE_URL': 'http://localhost:1234/', 'OLLAMA_TIMEOUT': '600'}, clear=True), patch.object(api.openai, 'OpenAI') as constructor:
            api.get_client()
            self.assertEqual(constructor.call_args.kwargs['base_url'], 'http://localhost:1234/v1')
            self.assertEqual(constructor.call_args.kwargs['timeout'], 600)

    def test_local_stop_and_truncation(self):
        self.client.chat.completions.create.return_value = self.response('working\npartial', 'length')
        self.assertEqual(api.GPT_request('prompt', {'stop': '\n'}), 'working')
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_truncation_retries_are_bounded(self):
        self.client.chat.completions.create.return_value = self.response('partial', 'length')
        with self.assertRaises(api.ModelResponseError):
            api.GPT_request('prompt', {'max_tokens': 5})
        self.assertEqual([c.kwargs['max_tokens'] for c in self.client.chat.completions.create.call_args_list], [1024, 2048, 4096])

    def test_empty_output_and_connection_errors_propagate(self):
        self.client.chat.completions.create.return_value = self.response('')
        with self.assertRaises(api.ModelResponseError):
            api.ChatGPT_request('prompt')
        self.client.chat.completions.create.side_effect = api.openai.OpenAIError('offline')
        with self.assertRaises(api.openai.OpenAIError):
            api.ChatGPT_safe_generate_response('prompt', 'example', 'instruction')

    def test_local_embedding_model(self):
        self.client.embeddings.create.return_value = types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1, 2])])
        self.assertEqual(api.get_embedding(''), [1, 2])
        self.client.embeddings.create.assert_called_once_with(input=['this is blank'], model='local-embed', encoding_format='float')

    def test_memory_conversion_and_reuse(self):
        old = {'memory': [1, 2]}
        with patch.object(api, 'get_embedding', return_value=[3, 4]) as embed:
            self.assertEqual(api.prepare_memory_embeddings(old, None), {'memory': [3, 4]})
            self.assertEqual(old, {'memory': [1, 2]})
            embed.assert_called_once_with('memory')
            embed.reset_mock()
            self.assertIs(api.prepare_memory_embeddings(old, api.EMBEDDING_IDENTITY), old)
            embed.assert_not_called()
            api.prepare_memory_embeddings(old, {'provider': 'ollama', 'model': 'other'})
            embed.assert_called_once_with('memory')

    def test_memory_save_reload_tracks_embedding_model(self):
        memory_path = path.parents[1] / 'memory_structures/associative_memory.py'
        memory_spec = importlib.util.spec_from_file_location('memory_under_test', memory_path)
        memory_module = importlib.util.module_from_spec(memory_spec)
        with patch.dict(sys.modules, {'global_methods': types.ModuleType('global_methods'),
                'persona.prompt_template.gpt_structure': api}):
            memory_spec.loader.exec_module(memory_module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'embeddings.json').write_text(json.dumps({'old memory': [1, 2]}))
            (root / 'nodes.json').write_text('{}')
            (root / 'kw_strength.json').write_text(json.dumps({
                'kw_strength_event': {}, 'kw_strength_thought': {}}))
            with patch.object(api, 'get_embedding', return_value=[3, 4]) as embed:
                memory = memory_module.AssociativeMemory(folder)
                self.assertEqual(memory.embeddings['old memory'], [3, 4])
                self.assertEqual(json.loads((root / 'embeddings.json').read_text())['old memory'], [1, 2])
                memory.save(folder)
                self.assertEqual(json.loads((root / 'embedding_metadata.json').read_text()), api.EMBEDDING_IDENTITY)
                embed.reset_mock()
                restored = memory_module.AssociativeMemory(folder)
                self.assertEqual(restored.embeddings, memory.embeddings)
                embed.assert_not_called()


if __name__ == '__main__':
    unittest.main()
