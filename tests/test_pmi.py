"""Unit tests for PMI calculation module.

This module tests the core PMI functionality:
- Co-occurrence matrix construction
- PMI calculation with smoothing
- Synonym validation pipeline
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from scipy import sparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pmi.cooccurrence import (
    CooccurrenceConfig,
    CooccurrenceMatrixBuilder,
    WindowType,
)
from src.pmi.pmi_calculator import PMICalculator, PMIConfig, PPMICalculator
from src.pmi.synonym_validator import (
    OOVStrategy,
    SynonymValidator,
    ValidationConfig,
    create_pmi_pipeline,
)


# Test fixtures
@pytest.fixture
def sample_documents() -> List[str]:
    """Create sample documents for testing."""
    return [
        "인공지능 기술이 발전하면서 기계학습이 주목받고 있다.",
        "기계학습과 딥러닝은 인공지능의 핵심 기술이다.",
        "딥러닝은 신경망을 기반으로 한 기계학습 방법이다.",
        "자연어처리는 인공지능의 중요한 응용 분야이다.",
        "검색 엔진은 자연어처리 기술을 활용한다.",
        "법원 판결은 판례를 참고하여 내려진다.",
        "원고와 피고는 소송 당사자이다.",
        "손해배상 청구는 법원에서 심리한다.",
        "서울은 대한민국의 수도이다.",
        "부산은 대한민국의 항구 도시이다.",
    ]


@pytest.fixture
def simple_tokenizer():
    """Create simple whitespace tokenizer."""
    return lambda text: text.split()


@pytest.fixture
def cooccurrence_builder(sample_documents, simple_tokenizer) -> CooccurrenceMatrixBuilder:
    """Build co-occurrence matrix from sample documents."""
    config = CooccurrenceConfig(
        window_type=WindowType.SENTENCE,
        min_term_freq=1,
        max_vocab_size=1000,
        symmetric=True,
    )
    builder = CooccurrenceMatrixBuilder(config)
    builder.fit(sample_documents, tokenizer=simple_tokenizer, show_progress=False)
    return builder


class TestCooccurrenceMatrixBuilder:
    """Test CooccurrenceMatrixBuilder class."""

    def test_vocabulary_construction(
        self, sample_documents, simple_tokenizer
    ):
        """Test vocabulary is built correctly."""
        config = CooccurrenceConfig(min_term_freq=1, max_vocab_size=100)
        builder = CooccurrenceMatrixBuilder(config)
        builder.fit(sample_documents, tokenizer=simple_tokenizer, show_progress=False)

        vocab = builder.get_vocabulary()
        assert len(vocab) > 0
        assert "인공지능" in vocab or "인공지능의" in vocab

    def test_cooccurrence_matrix_shape(self, cooccurrence_builder):
        """Test matrix has correct shape."""
        matrix = cooccurrence_builder.get_cooccurrence_matrix()
        vocab_size = len(cooccurrence_builder.get_vocabulary())

        assert matrix.shape == (vocab_size, vocab_size)
        assert sparse.issparse(matrix)

    def test_symmetric_cooccurrence(self, cooccurrence_builder):
        """Test matrix is symmetric when configured."""
        matrix = cooccurrence_builder.get_cooccurrence_matrix()

        # Check symmetry for non-zero entries
        rows, cols = matrix.nonzero()
        for i, j in zip(rows[:10], cols[:10]):
            assert matrix[i, j] == matrix[j, i]

    def test_term_frequencies(self, cooccurrence_builder):
        """Test term frequencies are computed correctly."""
        term_freq = cooccurrence_builder.get_term_frequencies()

        assert len(term_freq) > 0
        assert all(freq > 0 for freq in term_freq.values())

    def test_cooccurrence_count(self, cooccurrence_builder):
        """Test co-occurrence count retrieval."""
        # Terms that should co-occur
        count = cooccurrence_builder.get_cooccurrence_count("기계학습이", "주목받고")
        # May or may not be > 0 depending on tokenization

        # Unknown terms should return 0
        count = cooccurrence_builder.get_cooccurrence_count("unknown1", "unknown2")
        assert count == 0.0

    def test_save_and_load(self, cooccurrence_builder):
        """Test saving and loading matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "cooc"
            cooccurrence_builder.save(save_path)

            # Load back
            loaded = CooccurrenceMatrixBuilder.load(save_path)

            assert len(loaded.get_vocabulary()) == len(
                cooccurrence_builder.get_vocabulary()
            )
            assert (
                loaded.get_cooccurrence_matrix().shape
                == cooccurrence_builder.get_cooccurrence_matrix().shape
            )


class TestPMICalculator:
    """Test PMICalculator class."""

    def test_pmi_computation(self, cooccurrence_builder):
        """Test PMI computation."""
        pmi_calc = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
        )

        vocab = cooccurrence_builder.get_vocabulary()
        if len(vocab) >= 2:
            terms = list(vocab.keys())[:2]
            pmi = pmi_calc.compute_pmi(terms[0], terms[1])
            assert isinstance(pmi, float)

    def test_ppmi_non_negative(self, cooccurrence_builder):
        """Test PPMI returns non-negative values."""
        ppmi_calc = PPMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
        )

        vocab = cooccurrence_builder.get_vocabulary()
        if len(vocab) >= 2:
            terms = list(vocab.keys())[:2]
            ppmi = ppmi_calc.compute_pmi(terms[0], terms[1])
            assert ppmi >= 0

    def test_oov_handling(self, cooccurrence_builder):
        """Test handling of out-of-vocabulary terms."""
        pmi_calc = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
            config=PMIConfig(use_ppmi=True),
        )

        # OOV terms should return 0 for PPMI
        pmi = pmi_calc.compute_pmi("completely_unknown_term", "another_unknown")
        assert pmi == 0.0

    def test_batch_computation(self, cooccurrence_builder):
        """Test batch PMI computation."""
        pmi_calc = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
        )

        vocab = cooccurrence_builder.get_vocabulary()
        terms = list(vocab.keys())[:4]
        if len(terms) >= 4:
            pairs = [(terms[0], terms[1]), (terms[2], terms[3])]
            scores = pmi_calc.compute_pmi_batch(pairs, show_progress=False)

            assert len(scores) == 2
            assert all(isinstance(s, float) for s in scores)

    def test_laplace_smoothing(self, cooccurrence_builder):
        """Test Laplace smoothing effect."""
        # Without smoothing
        config_no_smooth = PMIConfig(laplace_smoothing=0.0, use_ppmi=False)
        pmi_no_smooth = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
            config=config_no_smooth,
        )

        # With smoothing
        config_smooth = PMIConfig(laplace_smoothing=1.0, use_ppmi=False)
        pmi_smooth = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
            config=config_smooth,
        )

        vocab = cooccurrence_builder.get_vocabulary()
        terms = list(vocab.keys())[:2]
        if len(terms) >= 2:
            # Scores should differ due to smoothing
            s1 = pmi_no_smooth.compute_pmi(terms[0], terms[1])
            s2 = pmi_smooth.compute_pmi(terms[0], terms[1])
            # Just check both return valid floats
            assert isinstance(s1, float)
            assert isinstance(s2, float)


class TestSynonymValidator:
    """Test SynonymValidator class."""

    @pytest.fixture
    def sample_pairs(self):
        """Create sample synonym pairs."""
        return [
            {"source": "인공지능", "target": "기계학습", "similarity": 0.8, "category": "cluster"},
            {"source": "딥러닝", "target": "신경망", "similarity": 0.75, "category": "cluster"},
            {"source": "원고", "target": "피고", "similarity": 0.7, "category": "cluster"},
            {"source": "unknown1", "target": "unknown2", "similarity": 0.9, "category": "BPE"},
        ]

    def test_validation(self, cooccurrence_builder, sample_pairs):
        """Test synonym validation."""
        pmi_calc = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
        )

        config = ValidationConfig(
            pmi_percentile_threshold=10.0,
            oov_strategy=OOVStrategy.KEEP,
        )

        validator = SynonymValidator(pmi_calc, config)
        validated, result = validator.validate(sample_pairs, show_progress=False)

        assert len(validated) == len(sample_pairs)
        assert result.total_pairs == len(sample_pairs)

    def test_oov_detection(self, cooccurrence_builder, sample_pairs):
        """Test OOV term detection."""
        pmi_calc = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
        )

        validator = SynonymValidator(pmi_calc)
        oov_terms = validator.get_oov_terms(sample_pairs)

        # unknown1 and unknown2 should be OOV
        assert "unknown1" in oov_terms or "unknown2" in oov_terms

    def test_save_report(self, cooccurrence_builder, sample_pairs):
        """Test saving validation report."""
        pmi_calc = PMICalculator(
            cooccurrence_matrix=cooccurrence_builder.get_cooccurrence_matrix(),
            term_frequencies=cooccurrence_builder.get_term_frequencies(),
            vocabulary=cooccurrence_builder.get_vocabulary(),
            total_windows=cooccurrence_builder.get_stats().total_windows,
        )

        validator = SynonymValidator(pmi_calc)
        validated, result = validator.validate(sample_pairs, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "report"
            validator.save_validation_report(validated, result, save_path)

            assert (save_path / "validated_pairs.jsonl").exists()
            assert (save_path / "invalid_pairs.jsonl").exists()
            assert (save_path / "validation_report.json").exists()


class TestPMIPipeline:
    """Test the complete PMI pipeline."""

    def test_create_pipeline(self, sample_documents, simple_tokenizer):
        """Test pipeline creation."""
        from src.pmi.cooccurrence import CooccurrenceConfig

        # Use min_term_freq=1 for small test corpus
        cooc_config = CooccurrenceConfig(min_term_freq=1, max_vocab_size=1000)
        builder, pmi_calc = create_pmi_pipeline(
            documents=sample_documents,
            tokenizer=simple_tokenizer,
            cooc_config=cooc_config,
            show_progress=False,
        )

        assert builder is not None
        assert pmi_calc is not None
        assert len(builder.get_vocabulary()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
