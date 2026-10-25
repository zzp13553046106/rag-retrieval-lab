"""Regression tests with tiny tensors and model stubs; no model downloads."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag


class TensorInputs(dict):
    @property
    def input_ids(self):
        return self['input_ids']


class GenerationTokenizer:
    eos_token_id = 0

    def __init__(self, answer):
        self.answer = answer

    def apply_chat_template(self, messages, **kwargs):
        return 'test prompt'

    def __call__(self, text, **kwargs):
        return TensorInputs(input_ids=torch.tensor([[1, 2]]))

    def decode(self, token_ids, **kwargs):
        return self.answer


class GenerationModel:
    def generate(self, input_ids, **kwargs):
        return torch.cat((input_ids, torch.tensor([[3]])), dim=1)


class GenerationRegressionTests(unittest.TestCase):
    def make_engine(self, answer):
        engine = rag.Engine.__new__(rag.Engine)
        engine.torch = torch
        engine.chunks = [
            {'id': 'doc.a', 'title': 'Document A', 'text': 'Available evidence.'},
        ]
        engine.gt = GenerationTokenizer(answer)
        engine.gm = GenerationModel()
        return engine

    def generate(self, answer):
        engine = self.make_engine(answer)
        return engine.generate('Question?', [0], {'doc.a': 1.0}, threshold=0.0)

    def test_empty_ranking_refuses_without_generator(self):
        engine = rag.Engine.__new__(rag.Engine)
        engine.chunks = []
        result = engine.generate('Question?', [], {}, threshold=0.0)
        self.assertIs(result['refused'], True)
        self.assertEqual(result['evidence'], [])

    def test_missing_citation_is_distinct_from_invalid_citation(self):
        result = self.generate('The answer has no citation.')
        self.assertIs(result['missing_citations'], True)
        self.assertEqual(result['cited_ids'], [])
        self.assertEqual(result['invalid_citations'], [])

    def test_valid_citation_is_present_and_not_invalid(self):
        result = self.generate('The answer uses evidence [doc.a].')
        self.assertIs(result['missing_citations'], False)
        self.assertEqual(result['cited_ids'], ['doc.a'])
        self.assertEqual(result['invalid_citations'], [])

    def test_invalid_citation_is_reported_without_marking_it_missing(self):
        result = self.generate('The answer cites unavailable evidence [doc.unknown].')
        self.assertIs(result['missing_citations'], False)
        self.assertEqual(result['cited_ids'], ['doc.unknown'])
        self.assertEqual(result['invalid_citations'], ['doc.unknown'])

    def test_refusal_policy_in_answer_body_is_not_a_refusal(self):
        result = self.generate('回答策略要求：资料不足时应拒答 [doc.a]。')
        self.assertIs(result['refused'], False)
        self.assertIs(result['missing_citations'], False)
        self.assertEqual(result['cited_ids'], ['doc.a'])
        self.assertEqual(result['invalid_citations'], [])

    def test_explicit_refusal_does_not_require_citation(self):
        result = self.generate('资料不足。')
        self.assertIs(result['refused'], True)
        self.assertIs(result['missing_citations'], False)
        self.assertEqual(result['cited_ids'], [])
        self.assertEqual(result['invalid_citations'], [])


class TokenLimitRegressionTests(unittest.TestCase):
    def test_embedding_tokenizer_receives_configured_limit(self):
        engine = rag.Engine.__new__(rag.Engine)
        engine.torch = torch
        engine.et = Mock(return_value={'input_ids': torch.tensor([[1, 2]])})
        engine.em = Mock(return_value=SimpleNamespace(
            last_hidden_state=torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])))

        with patch.dict(rag.CONFIG, {'embedding_max_tokens': 7}):
            vectors = engine.encode(['A document longer than the chosen token limit.'])

        self.assertEqual(engine.et.call_args.kwargs['max_length'], 7)
        self.assertEqual(vectors.shape, (1, 2))
        torch.testing.assert_close(torch.from_numpy(vectors), torch.tensor([[0.6, 0.8]]))

    def test_reranker_tokenizer_receives_configured_limit(self):
        engine = rag.Engine.__new__(rag.Engine)
        engine.torch = torch
        engine.chunks = [{'id': 'doc.a'}, {'id': 'doc.b'}]
        engine.texts = ['First document.', 'Second document.']
        engine.rt = Mock(return_value={'input_ids': torch.tensor([[1, 2], [3, 4]])})
        engine.rm = Mock(return_value=SimpleNamespace(logits=torch.tensor([[0.0], [2.0]])))

        with patch.dict(rag.CONFIG, {'reranker_max_tokens': 11}):
            ranked, scores = engine.rerank('Question?', [0, 1])

        self.assertEqual(engine.rt.call_args.kwargs['max_length'], 11)
        self.assertEqual(ranked, [1, 0])
        self.assertEqual(scores, {'doc.b': 2.0, 'doc.a': 0.0})


class CalibrationRegressionTests(unittest.TestCase):
    def test_empty_calibration_data_raises_value_error(self):
        with self.assertRaises(ValueError):
            rag.calibrate([])

    def test_only_answerable_calibration_data_raises_value_error(self):
        rows = [
            {'top_score': 1.0, 'relevant_ids': ['doc.a']},
            {'top_score': 2.0, 'relevant_ids': ['doc.b']},
        ]
        with self.assertRaises(ValueError):
            rag.calibrate(rows)

    def test_only_unanswerable_calibration_data_raises_value_error(self):
        rows = [
            {'top_score': -1.0, 'relevant_ids': []},
            {'top_score': -2.0, 'relevant_ids': []},
        ]
        with self.assertRaises(ValueError):
            rag.calibrate(rows)

    def test_nonfinite_calibration_scores_raise_value_error(self):
        for score in (float('nan'), float('inf'), float('-inf')):
            for invalid_row in (0, 1):
                with self.subTest(score=score, invalid_row=invalid_row):
                    rows = [
                        {'top_score': 1.0, 'relevant_ids': ['doc.a']},
                        {'top_score': -1.0, 'relevant_ids': []},
                    ]
                    rows[invalid_row]['top_score'] = score
                    with self.assertRaises(ValueError):
                        rag.calibrate(rows)


class InputValidationRegressionTests(unittest.TestCase):
    def test_rank_rejects_negative_limit_and_accepts_zero(self):
        with self.subTest(k=-1):
            with self.assertRaises(ValueError):
                rag.rank([1.0, 2.0], -1)
        with self.subTest(k=0):
            self.assertEqual(rag.rank([1.0, 2.0], 0), [])

    def test_rrf_rejects_negative_or_nonfinite_constant(self):
        for constant in (-1, float('nan')):
            with self.subTest(constant=constant):
                with self.assertRaises(ValueError):
                    rag.rrf([[0, 1], [1, 0]], constant=constant)

    def test_default_config_is_valid(self):
        rag.validate_config()

    def test_config_rejects_unsupported_or_nonpositive_values(self):
        invalid_values = [
            ('candidate_k', 0),
            ('context_k', 0),
            ('threads', 0),
            ('embedding_max_tokens', 0),
            ('embedding_max_tokens', -1),
            ('reranker_max_tokens', 0),
            ('reranker_max_tokens', -1),
            ('device', 'cuda'),
        ]
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                with patch.dict(rag.CONFIG, {key: value}):
                    with self.assertRaises(ValueError):
                        rag.validate_config()

    def test_whitespace_query_rejected_before_engine_access(self):
        engine = rag.Engine.__new__(rag.Engine)
        with self.assertRaises(ValueError):
            engine.search('   ')


if __name__ == '__main__':
    unittest.main()
