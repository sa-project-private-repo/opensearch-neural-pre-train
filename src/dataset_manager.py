"""
데이터셋 저장 및 로드 유틸리티

노트북 간 데이터 공유를 위한 DatasetManager 클래스를 제공합니다.
JSON, Pickle, PyTorch 모델 등 다양한 형식의 데이터를 저장하고 로드할 수 있습니다.

Usage:
    from src.dataset_manager import DatasetManager

    # 초기화
    dm = DatasetManager(base_path="dataset")

    # JSON 저장/로드
    dm.save_json({"key": "value"}, "data.json", "base_model")
    data = dm.load_json("data.json", "base_model")

    # Pickle 저장/로드
    dm.save_pickle(my_object, "data.pkl", "base_model")
    obj = dm.load_pickle("data.pkl", "base_model")

    # 모델 저장/로드
    dm.save_model(model, tokenizer, "my_model", "base_model")
    model, tokenizer = dm.load_model(ModelClass, "my_model", "base_model")
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class DatasetManager:
    """노트북 간 데이터 공유를 위한 매니저"""

    def __init__(self, base_path: str = "dataset"):
        """
        DatasetManager 초기화

        Args:
            base_path: 데이터 저장 기본 경로 (기본값: "dataset")
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # 메타데이터 파일 경로
        self.metadata_path = self.base_path / "metadata.json"

        # 메타데이터 초기화 또는 로드
        self._init_metadata()

    def _init_metadata(self):
        """메타데이터 초기화 또는 로드"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "datasets": {}
            }
            self._save_metadata()

    def _save_metadata(self):
        """메타데이터 저장"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _update_dataset_metadata(self, subdir: str, filename: str, info: Dict[str, Any]):
        """데이터셋 메타데이터 업데이트"""
        key = f"{subdir}/{filename}" if subdir else filename
        self.metadata["datasets"][key] = {
            **info,
            "updated_at": datetime.now().isoformat()
        }
        self._save_metadata()

    def save_json(self, data: Any, filename: str, subdir: str = "") -> Path:
        """
        JSON 형식으로 데이터 저장

        Args:
            data: 저장할 데이터 (JSON 직렬화 가능해야 함)
            filename: 파일명
            subdir: 하위 디렉토리 (선택)

        Returns:
            저장된 파일의 경로
        """
        path = self.base_path / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved JSON: {path}")

        # 메타데이터 업데이트
        self._update_dataset_metadata(subdir, filename, {
            "type": "json",
            "size_bytes": path.stat().st_size
        })

        return path

    def load_json(self, filename: str, subdir: str = "") -> Any:
        """
        JSON 파일 로드

        Args:
            filename: 파일명
            subdir: 하위 디렉토리 (선택)

        Returns:
            로드된 데이터

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
        """
        path = self.base_path / subdir / filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✓ Loaded JSON: {path}")
        return data

    def save_pickle(self, data: Any, filename: str, subdir: str = "") -> Path:
        """
        Pickle 형식으로 데이터 저장

        Args:
            data: 저장할 Python 객체
            filename: 파일명
            subdir: 하위 디렉토리 (선택)

        Returns:
            저장된 파일의 경로
        """
        path = self.base_path / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Saved Pickle: {path}")

        # 메타데이터 업데이트
        self._update_dataset_metadata(subdir, filename, {
            "type": "pickle",
            "size_bytes": path.stat().st_size
        })

        return path

    def load_pickle(self, filename: str, subdir: str = "") -> Any:
        """
        Pickle 파일 로드

        Args:
            filename: 파일명
            subdir: 하위 디렉토리 (선택)

        Returns:
            로드된 Python 객체

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
        """
        path = self.base_path / subdir / filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Loaded Pickle: {path}")
        return data

    def save_model(
        self,
        model,
        tokenizer,
        model_dir: str,
        subdir: str = ""
    ) -> Path:
        """
        PyTorch 모델 저장 (Hugging Face 및 일반 PyTorch 모델 지원)

        Args:
            model: 저장할 모델 (Hugging Face 또는 일반 PyTorch 모델)
            tokenizer: 저장할 토크나이저
            model_dir: 모델 디렉토리명
            subdir: 하위 디렉토리 (선택)

        Returns:
            저장된 모델 디렉토리 경로
        """
        import torch
        import json

        path = self.base_path / subdir / model_dir
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(model, 'save_pretrained'):
            # Hugging Face 모델
            model.save_pretrained(path)
        else:
            # 일반 PyTorch 모델
            torch.save(model.state_dict(), path / "pytorch_model.bin")

            # Config 저장 (모델 구조 정보)
            config = {
                "model_type": model.__class__.__name__,
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            }

            # 모델에 config 속성이 있으면 추가
            if hasattr(model, 'config'):
                if hasattr(model.config, 'to_dict'):
                    config.update(model.config.to_dict())
                else:
                    config['model_config'] = str(model.config)

            # vocab_size 등 기본 속성 저장
            if hasattr(model, 'vocab_size'):
                config['vocab_size'] = model.vocab_size

            # BERT 기반 모델의 경우 base model 정보 저장
            if hasattr(model, 'bert'):
                if hasattr(model.bert, 'config'):
                    config['base_model_name'] = model.bert.config.name_or_path if hasattr(model.bert.config, 'name_or_path') else 'unknown'

            with open(path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        # Save tokenizer
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(path)
        else:
            # 일반 토크나이저인 경우 pickle로 저장
            import pickle
            with open(path / "tokenizer.pkl", 'wb') as f:
                pickle.dump(tokenizer, f)

        print(f"✓ Saved Model: {path}")

        # 메타데이터 업데이트
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        self._update_dataset_metadata(subdir, model_dir, {
            "type": "pytorch_model",
            "size_bytes": total_size
        })

        return path

    def load_model(
        self,
        model_class,
        model_dir: str,
        subdir: str = "",
        device: str = "cpu"
    ) -> Tuple[Any, Any]:
        """
        PyTorch 모델 로드 (Hugging Face 및 일반 PyTorch 모델 지원)

        Args:
            model_class: 모델 클래스 (예: OpenSearchSparseEncoder)
            model_dir: 모델 디렉토리명
            subdir: 하위 디렉토리 (선택)
            device: 로드할 디바이스 ("cpu", "cuda", etc.)

        Returns:
            (model, tokenizer) 튜플

        Raises:
            FileNotFoundError: 모델이 존재하지 않을 때
        """
        import torch
        import json

        path = self.base_path / subdir / model_dir

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        # Load model
        if hasattr(model_class, 'from_pretrained'):
            # Hugging Face 모델
            model = model_class.from_pretrained(path)
        else:
            # 일반 PyTorch 모델
            # Config 로드
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Config에서 필요한 인자를 추출하여 모델 초기화
                try:
                    # base_model_name이 있으면 전달 (OpenSearchDocEncoder 등)
                    if 'base_model_name' in config:
                        model = model_class(model_name=config['base_model_name'])
                    # vocab_size만 있는 경우
                    elif 'vocab_size' in config:
                        model = model_class(vocab_size=config['vocab_size'])
                    else:
                        model = model_class()
                except TypeError as e:
                    # 인자가 맞지 않으면 기본 초기화 시도
                    print(f"⚠️  Failed to initialize with config: {e}")
                    print(f"   Trying default initialization...")
                    model = model_class()
            else:
                # Config 없으면 기본 초기화
                model = model_class()

            # State dict 로드
            model_path = path / "pytorch_model.bin"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
            else:
                raise FileNotFoundError(f"Model weights not found: {model_path}")

        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception:
            # Pickle로 저장된 토크나이저 로드
            import pickle
            tokenizer_path = path / "tokenizer.pkl"
            if tokenizer_path.exists():
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            else:
                raise FileNotFoundError(f"Tokenizer not found in {path}")

        print(f"✓ Loaded Model: {path}")
        return model, tokenizer

    def check_data_exists(self, filename: str, subdir: str = "") -> bool:
        """
        데이터 파일/디렉토리 존재 확인

        Args:
            filename: 파일명 또는 디렉토리명
            subdir: 하위 디렉토리 (선택)

        Returns:
            존재하면 True, 아니면 False
        """
        path = self.base_path / subdir / filename
        return path.exists()

    def list_files(self, subdir: str = "") -> List[str]:
        """
        특정 디렉토리의 파일 목록 조회

        Args:
            subdir: 하위 디렉토리 (선택)

        Returns:
            파일명 리스트
        """
        path = self.base_path / subdir
        if not path.exists():
            return []

        return [f.name for f in path.iterdir() if f.is_file()]

    def list_directories(self, subdir: str = "") -> List[str]:
        """
        특정 디렉토리의 하위 디렉토리 목록 조회

        Args:
            subdir: 하위 디렉토리 (선택)

        Returns:
            디렉토리명 리스트
        """
        path = self.base_path / subdir
        if not path.exists():
            return []

        return [d.name for d in path.iterdir() if d.is_dir()]

    def check_dependencies(self, required: List[Tuple[str, str]]) -> bool:
        """
        노트북 실행 전 필요한 데이터 파일 확인

        Args:
            required: [(subdir, filename), ...] 형식의 필수 파일 리스트

        Returns:
            모든 파일이 존재하면 True, 아니면 False

        Example:
            >>> required = [
            ...     ("base_model", "korean_documents.json"),
            ...     ("base_model", "qd_pairs_base.pkl"),
            ... ]
            >>> dm.check_dependencies(required)
        """
        missing = []
        for subdir, filename in required:
            if not self.check_data_exists(filename, subdir):
                missing.append(f"{subdir}/{filename}")

        if missing:
            print("=" * 70)
            print("❌ Missing required data files:")
            print("=" * 70)
            for f in missing:
                print(f"   - {f}")
            print("\n💡 Please run previous notebooks first:")
            print("   1. 01_neural_sparse_base_training.ipynb")
            print("   2. 02_llm_synthetic_data_generation.ipynb")
            print("=" * 70)
            return False

        print("✅ All dependencies satisfied")
        return True

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        저장된 모든 데이터셋 정보 조회

        Returns:
            데이터셋 메타데이터 딕셔너리
        """
        return self.metadata

    def print_summary(self):
        """데이터셋 요약 정보 출력"""
        print("=" * 70)
        print("📊 Dataset Summary")
        print("=" * 70)
        print(f"Base path: {self.base_path.absolute()}")
        print(f"Total datasets: {len(self.metadata['datasets'])}")

        # 하위 디렉토리별 그룹화
        by_subdir = {}
        for key, info in self.metadata['datasets'].items():
            if '/' in key:
                subdir = key.split('/')[0]
            else:
                subdir = "root"

            if subdir not in by_subdir:
                by_subdir[subdir] = []
            by_subdir[subdir].append((key, info))

        print("\nDatasets by directory:")
        for subdir, datasets in sorted(by_subdir.items()):
            print(f"\n  📁 {subdir}/")
            total_size = 0
            for key, info in datasets:
                filename = key.split('/')[-1]
                size_mb = info.get('size_bytes', 0) / 1024 / 1024
                dtype = info.get('type', 'unknown')
                print(f"     - {filename:<40} ({dtype:>15}, {size_mb:>6.1f} MB)")
                total_size += info.get('size_bytes', 0)

            print(f"     {'Total:':<40} {'':<15}  {total_size/1024/1024:>6.1f} MB")

        print("=" * 70)

    def clear_subdirectory(self, subdir: str, confirm: bool = False):
        """
        특정 하위 디렉토리의 모든 데이터 삭제

        Args:
            subdir: 삭제할 하위 디렉토리
            confirm: 확인 없이 삭제 (기본값: False)

        Warning:
            이 작업은 되돌릴 수 없습니다!
        """
        path = self.base_path / subdir

        if not path.exists():
            print(f"⚠️  Directory not found: {path}")
            return

        if not confirm:
            print(f"⚠️  WARNING: This will delete all data in {path}")
            print("   To confirm, call with confirm=True")
            return

        import shutil
        shutil.rmtree(path)

        # 메타데이터에서 제거
        keys_to_remove = [k for k in self.metadata['datasets'].keys() if k.startswith(f"{subdir}/")]
        for key in keys_to_remove:
            del self.metadata['datasets'][key]

        self._save_metadata()
        print(f"✓ Cleared directory: {path}")
