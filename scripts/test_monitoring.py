#!/usr/bin/env python3
"""
Test monitoring system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring import SystemMonitor

def test_basic_monitoring():
    """기본 모니터링 테스트"""
    print("="*70)
    print("🧪 Testing System Monitoring")
    print("="*70)

    monitor = SystemMonitor()

    print("\n1️⃣ Testing GPU metrics...")
    gpu_metrics = monitor.get_gpu_metrics()
    if gpu_metrics:
        print(f"   ✅ Found {len(gpu_metrics)} GPU(s)")
        for gpu in gpu_metrics:
            print(f"      - {gpu['name']}: {gpu['memory_util']:.1f}% memory used")
    else:
        print("   ⚠️  No GPU detected or GPUtil not installed")

    print("\n2️⃣ Testing CPU metrics...")
    cpu_metrics = monitor.get_cpu_metrics()
    if cpu_metrics:
        print(f"   ✅ CPU: {cpu_metrics.get('cpu_percent', 0):.1f}% usage")
        print(f"      Cores: {cpu_metrics.get('cpu_count_physical', 'N/A')} physical")
    else:
        print("   ❌ Failed to get CPU metrics")

    print("\n3️⃣ Testing memory metrics...")
    mem_metrics = monitor.get_memory_metrics()
    if mem_metrics:
        print(f"   ✅ Memory: {mem_metrics.get('used_gb', 0):.1f}/"
              f"{mem_metrics.get('total_gb', 0):.1f} GB "
              f"({mem_metrics.get('percent', 0):.1f}%)")
    else:
        print("   ❌ Failed to get memory metrics")

    print("\n4️⃣ Testing disk metrics...")
    disk_metrics = monitor.get_disk_metrics()
    if disk_metrics:
        print(f"   ✅ Disk: {disk_metrics.get('used_gb', 0):.1f}/"
              f"{disk_metrics.get('total_gb', 0):.1f} GB "
              f"({disk_metrics.get('percent', 0):.1f}%)")
    else:
        print("   ❌ Failed to get disk metrics")

    print("\n5️⃣ Testing full metrics collection...")
    try:
        all_metrics = monitor.get_all_metrics()
        print(f"   ✅ Collected metrics with {len(all_metrics)} categories")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    print("\n6️⃣ Testing log file writing...")
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        log_file = f.name

    try:
        monitor_with_log = SystemMonitor(log_file=log_file)
        monitor_with_log.get_all_metrics()

        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            print(f"   ✅ Log file created: {log_file}")
            print(f"      Size: {os.path.getsize(log_file)} bytes")
        else:
            print("   ❌ Log file not created")

        # Cleanup
        os.unlink(log_file)
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    print("\n" + "="*70)
    print("📊 Displaying full summary...")
    print("="*70)

    monitor.print_summary(clear_screen=False)

    print("\n" + "="*70)
    print("✅ All tests complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run live monitoring: python -m src.monitoring.system_monitor")
    print("  2. In notebook: from src.monitoring import monitor_training")
    print("  3. Monitor training: monitor_training(interval=5)")
    print("")


if __name__ == "__main__":
    test_basic_monitoring()
