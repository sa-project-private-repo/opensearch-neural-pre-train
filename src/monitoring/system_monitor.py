"""
System monitoring utilities for ML training
"""

import time
import psutil
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path


class SystemMonitor:
    """실시간 시스템 메트릭 수집"""

    def __init__(self, log_file: Optional[str] = None):
        """
        SystemMonitor 초기화

        Args:
            log_file: 메트릭 로그 파일 경로 (선택)
        """
        self.log_file = log_file
        self.start_time = time.time()

        # GPU 라이브러리 동적 import
        self.gpu_available = False
        try:
            import GPUtil
            self.GPUtil = GPUtil
            self.gpu_available = True
        except ImportError:
            print("⚠️  GPUtil not installed. GPU metrics will not be available.")
            print("   Install with: pip install gputil")

    def get_gpu_metrics(self) -> List[Dict]:
        """GPU 메트릭 수집"""
        if not self.gpu_available:
            return []

        try:
            gpus = self.GPUtil.getGPUs()
            metrics = []

            for gpu in gpus:
                metrics.append({
                    'gpu_id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # %
                    'memory_used': gpu.memoryUsed,  # MB
                    'memory_total': gpu.memoryTotal,  # MB
                    'memory_util': gpu.memoryUtil * 100,  # %
                    'memory_free': gpu.memoryFree,  # MB
                    'temperature': gpu.temperature,  # °C
                })

            return metrics
        except Exception as e:
            print(f"⚠️  Error getting GPU metrics: {e}")
            return []

    def get_cpu_metrics(self) -> Dict:
        """CPU 메트릭 수집"""
        try:
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            freq_current = cpu_freq.current if cpu_freq else None

            # Load average (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have getloadavg
                pass

            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_freq_current': freq_current,
                'load_avg_1m': load_avg[0] if load_avg else None,
                'load_avg_5m': load_avg[1] if load_avg else None,
                'load_avg_15m': load_avg[2] if load_avg else None,
            }
        except Exception as e:
            print(f"⚠️  Error getting CPU metrics: {e}")
            return {}

    def get_memory_metrics(self) -> Dict:
        """메모리 메트릭 수집"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                'total_gb': mem.total / (1024**3),  # GB
                'available_gb': mem.available / (1024**3),
                'used_gb': mem.used / (1024**3),
                'percent': mem.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent,
            }
        except Exception as e:
            print(f"⚠️  Error getting memory metrics: {e}")
            return {}

    def get_disk_metrics(self, path: str = '/') -> Dict:
        """디스크 메트릭 수집"""
        try:
            disk = psutil.disk_usage(path)
            io = psutil.disk_io_counters()

            return {
                'total_gb': disk.total / (1024**3),  # GB
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': disk.percent,
                'read_mb': io.read_bytes / (1024**2) if io else None,
                'write_mb': io.write_bytes / (1024**2) if io else None,
                'read_count': io.read_count if io else None,
                'write_count': io.write_count if io else None,
            }
        except Exception as e:
            print(f"⚠️  Error getting disk metrics: {e}")
            return {}

    def get_network_metrics(self) -> Dict:
        """네트워크 메트릭 수집"""
        try:
            net = psutil.net_io_counters()

            return {
                'bytes_sent_mb': net.bytes_sent / (1024**2),  # MB
                'bytes_recv_mb': net.bytes_recv / (1024**2),
                'packets_sent': net.packets_sent,
                'packets_recv': net.packets_recv,
                'errin': net.errin,
                'errout': net.errout,
                'dropin': net.dropin,
                'dropout': net.dropout,
            }
        except Exception as e:
            print(f"⚠️  Error getting network metrics: {e}")
            return {}

    def get_process_info(self) -> Dict:
        """현재 프로세스 정보"""
        try:
            process = psutil.Process()
            with process.oneshot():
                return {
                    'pid': process.pid,
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / (1024**2),
                    'num_threads': process.num_threads(),
                }
        except Exception as e:
            print(f"⚠️  Error getting process info: {e}")
            return {}

    def get_all_metrics(self) -> Dict:
        """모든 메트릭 수집"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'gpu': self.get_gpu_metrics(),
            'cpu': self.get_cpu_metrics(),
            'memory': self.get_memory_metrics(),
            'disk': self.get_disk_metrics(),
            'network': self.get_network_metrics(),
            'process': self.get_process_info(),
        }

        # 로그 파일에 저장
        if self.log_file:
            try:
                log_path = Path(self.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"⚠️  Error writing to log file: {e}")

        return metrics

    def print_summary(self, clear_screen: bool = True):
        """메트릭 요약 출력"""
        if clear_screen:
            # Clear screen (Unix/Windows compatible)
            import os
            os.system('clear' if os.name == 'posix' else 'cls')

        metrics = self.get_all_metrics()

        print("="*70)
        print(f"📊 System Metrics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # GPU
        if metrics['gpu']:
            print("\n🎮 GPU:")
            for gpu in metrics['gpu']:
                print(f"  [{gpu['gpu_id']}] {gpu['name']}")
                print(f"      Load: {gpu['load']:.1f}%")
                print(f"      Memory: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB "
                      f"({gpu['memory_util']:.1f}%) | Free: {gpu['memory_free']:.0f} MB")
                print(f"      Temp: {gpu['temperature']:.1f}°C")
        else:
            print("\n🎮 GPU: Not available")

        # CPU
        cpu = metrics['cpu']
        print(f"\n💻 CPU:")
        print(f"  Usage: {cpu.get('cpu_percent', 0):.1f}%")
        print(f"  Cores: {cpu.get('cpu_count_physical', 'N/A')} physical, "
              f"{cpu.get('cpu_count_logical', 'N/A')} logical")
        if cpu.get('cpu_freq_current'):
            print(f"  Frequency: {cpu['cpu_freq_current']:.0f} MHz")
        if cpu.get('load_avg_1m') is not None:
            print(f"  Load Avg: {cpu['load_avg_1m']:.2f} / {cpu['load_avg_5m']:.2f} / "
                  f"{cpu['load_avg_15m']:.2f}")

        # Memory
        mem = metrics['memory']
        print(f"\n🧠 Memory:")
        print(f"  Used: {mem.get('used_gb', 0):.1f}/{mem.get('total_gb', 0):.1f} GB "
              f"({mem.get('percent', 0):.1f}%)")
        print(f"  Available: {mem.get('available_gb', 0):.1f} GB")
        if mem.get('swap_total_gb', 0) > 0:
            print(f"  Swap: {mem['swap_used_gb']:.1f}/{mem['swap_total_gb']:.1f} GB "
                  f"({mem['swap_percent']:.1f}%)")

        # Disk
        disk = metrics['disk']
        print(f"\n💾 Disk:")
        print(f"  Used: {disk.get('used_gb', 0):.1f}/{disk.get('total_gb', 0):.1f} GB "
              f"({disk.get('percent', 0):.1f}%)")
        print(f"  Free: {disk.get('free_gb', 0):.1f} GB")
        if disk.get('read_mb') is not None:
            print(f"  I/O: Read {disk['read_mb']:.0f} MB, Write {disk['write_mb']:.0f} MB")

        # Network
        net = metrics['network']
        print(f"\n🌐 Network:")
        print(f"  Sent: {net.get('bytes_sent_mb', 0):.1f} MB "
              f"({net.get('packets_sent', 0):,} packets)")
        print(f"  Recv: {net.get('bytes_recv_mb', 0):.1f} MB "
              f"({net.get('packets_recv', 0):,} packets)")
        if net.get('errin', 0) > 0 or net.get('errout', 0) > 0:
            print(f"  Errors: In {net['errin']}, Out {net['errout']}")

        # Process
        proc = metrics['process']
        print(f"\n🔧 Current Process:")
        print(f"  PID: {proc.get('pid', 'N/A')}")
        print(f"  CPU: {proc.get('cpu_percent', 0):.1f}%")
        print(f"  Memory: {proc.get('memory_mb', 0):.0f} MB")
        print(f"  Threads: {proc.get('num_threads', 'N/A')}")

        print("="*70)


def monitor_training(
    interval: int = 5,
    log_file: str = "logs/training_metrics.jsonl",
    clear_screen: bool = True
):
    """
    학습 중 실시간 모니터링

    Args:
        interval: 업데이트 간격 (초)
        log_file: 로그 파일 경로
        clear_screen: 화면 클리어 여부

    Usage:
        In notebook or script:
        ```python
        from src.monitoring import monitor_training
        monitor_training(interval=5)
        ```

        Or run standalone:
        ```bash
        python -m src.monitoring.system_monitor
        ```
    """
    monitor = SystemMonitor(log_file=log_file)

    print(f"🔍 Monitoring started (interval: {interval}s)")
    print(f"📝 Logging to: {log_file}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            monitor.print_summary(clear_screen=clear_screen)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        print(f"📊 Metrics saved to: {log_file}")


if __name__ == "__main__":
    # Standalone monitoring
    import argparse

    parser = argparse.ArgumentParser(description='System monitoring for ML training')
    parser.add_argument('--interval', type=int, default=5,
                        help='Update interval in seconds (default: 5)')
    parser.add_argument('--log-file', type=str, default='logs/training_metrics.jsonl',
                        help='Log file path (default: logs/training_metrics.jsonl)')
    parser.add_argument('--no-clear', action='store_true',
                        help='Do not clear screen between updates')

    args = parser.parse_args()

    monitor_training(
        interval=args.interval,
        log_file=args.log_file,
        clear_screen=not args.no_clear
    )
