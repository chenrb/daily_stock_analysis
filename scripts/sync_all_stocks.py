#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同步整个A股市场所有股票数据

使用项目现有的数据获取逻辑，批量同步A股市场所有股票数据。

用法:
    # 获取股票列表并同步（默认从2025-12-01至今）
    python scripts/sync_all_stocks.py

    # 只获取股票列表（不同步数据）
    python scripts/sync_all_stocks.py --list-only

    # 自定义日期范围
    python scripts/sync_all_stocks.py --start 2025-01-01 --end 2025-12-31

    # 限制同步数量（测试用）
    python scripts/sync_all_stocks.py --limit 100

    # 过滤市场（主板/创业板/科创板等）
    python scripts/sync_all_stocks.py --market 创业板

    # 从断点续传（已有数据跳过）
    python scripts/sync_all_stocks.py --resume

    # 调整并发数
    python scripts/sync_all_stocks.py --workers 5

    # ===== Tushare 高效模式（推荐）=====
    # 按交易日批量获取，效率提升 20+ 倍
    # 5000 只股票只需 220 次请求（按交易日）vs 5000+ 次请求（按股票）
    python scripts/sync_all_stocks.py --by-date

    # Tushare 模式 + 限制日期范围
    python scripts/sync_all_stocks.py --by-date --start 2025-01-01 --end 2025-01-31
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import akshare as ak
import pandas as pd

from src.config import setup_env, get_config

setup_env()

from src.storage import DatabaseManager, get_db
from data_provider import DataFetcherManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
)
logger = logging.getLogger(__name__)


def get_all_a_stocks_from_akshare() -> pd.DataFrame:
    """
    使用 Akshare 获取A股所有股票列表

    Returns:
        DataFrame with columns: code, name, market
    """
    logger.info("正在获取A股市场所有股票列表...")

    try:
        # 获取沪深A股列表
        df = ak.stock_info_a_code_name()

        if df is not None and not df.empty:
            # 重命名列
            df = df.rename(columns={'code': 'code', 'name': 'name'})

            # 确定市场
            def get_market(code: str) -> str:
                if code.startswith('6'):
                    return '沪市主板'
                elif code.startswith('0'):
                    return '深市主板'
                elif code.startswith('3'):
                    return '创业板'
                elif code.startswith('688'):
                    return '科创板'
                else:
                    return '其他'

            df['market'] = df['code'].apply(get_market)

            logger.info(f"获取股票列表成功: {len(df)} 只")
            logger.info(f"分布情况:")
            for market, count in df['market'].value_counts().items():
                logger.info(f"  - {market}: {count} 只")

            return df
        else:
            logger.error("获取股票列表失败: 返回数据为空")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return pd.DataFrame()


def filter_stocks(
        df: pd.DataFrame,
        market_filter: str = None,
        limit: int = None
) -> List[str]:
    """
    过滤股票列表

    Args:
        df: 股票列表 DataFrame
        market_filter: 市场过滤（如：主板、创业板、科创板）
        limit: 限制数量

    Returns:
        股票代码列表
    """
    if df.empty:
        return []

    # 市场过滤
    if market_filter:
        df = df[df['market'].str.contains(market_filter, na=False)]
        logger.info(f"过滤后股票数量: {len(df)}")

    # 限制数量
    if limit and limit > 0:
        df = df.head(limit)
        logger.info(f"限制数量后: {len(df)} 只")

    return df['code'].tolist()


class StockDataSyncer:
    """股票数据同步器"""

    def __init__(
            self,
            db: DatabaseManager = None,
            max_workers: int = 3,
            force_refresh: bool = False
    ):
        """
        初始化同步器

        Args:
            db: 数据库管理器
            max_workers: 最大并发数
            force_refresh: 是否强制刷新（忽略本地缓存）
        """
        self.db = db or get_db()
        self.max_workers = max_workers
        self.force_refresh = force_refresh
        self.fetcher_manager = DataFetcherManager()

        # 统计信息
        self.total_stocks = 0
        self.success_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.failed_stocks = []

        logger.info(f"同步器初始化完成，并发数: {max_workers}")

    def fetch_and_save_stock_data(
            self,
            code: str,
            start_date: str,
            end_date: str
    ) -> tuple:
        """
        获取并保存单只股票数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Tuple[是否成功, 错误信息]
        """
        try:
            # 断点续传检查：如果今日数据已存在且不强制刷新，则跳过
            today = date.today()
            if not self.force_refresh and self.db.has_today_data(code, today):
                logger.debug(f"[{code}] 今日数据已存在，跳过获取（断点续传）")
                return True, "skipped"

            # 从数据源获取数据
            logger.debug(f"[{code}] 开始从数据源获取数据...")
            df, source_name = self.fetcher_manager.get_daily_data(
                stock_code=code,
                start_date=start_date,
                end_date=end_date,
                days=90  # 获取90天数据
            )

            if df is None or df.empty:
                return False, "获取数据为空"

            # 保存到数据库
            saved_count = self.db.save_daily_data(df, code, source_name)
            logger.info(f"[{code}] 数据保存成功（来源: {source_name}，新增/更新 {saved_count} 条）")

            return True, "success"

        except Exception as e:
            error_msg = f"获取/保存数据失败: {str(e)}"
            logger.error(f"[{code}] {error_msg}")
            return False, error_msg

    def sync_stocks(
            self,
            stock_codes: List[str],
            start_date: str = None,
            end_date: str = None,
            progress_interval: int = 10
    ) -> dict:
        """
        批量同步股票数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            progress_interval: 进度报告间隔

        Returns:
            同步统计信息
        """
        if not stock_codes:
            logger.warning("股票代码列表为空")
            return {}

        # 默认日期范围
        if end_date is None:
            end_date = date.today().strftime('%Y-%m-%d')
        if start_date is None:
            # 默认从2025-12-01开始
            start_date = '2025-9-01'

        self.total_stocks = len(stock_codes)
        logger.info(f"开始同步 {self.total_stocks} 只股票数据")
        logger.info(f"日期范围: {start_date} ~ {end_date}")

        start_time = time.time()
        last_progress_time = start_time

        # 并发获取数据
        results = {}
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(
                    self.fetch_and_save_stock_data,
                    code,
                    start_date,
                    end_date
                ): code
                for code in stock_codes
            }

            # 收集结果
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                completed_count += 1

                try:
                    success, message = future.result()
                    results[code] = {'success': success, 'message': message}

                    if success:
                        if message == "skipped":
                            self.skipped_count += 1
                        else:
                            self.success_count += 1
                    else:
                        self.failed_count += 1
                        self.failed_stocks.append(code)

                except Exception as e:
                    logger.error(f"[{code}] 执行异常: {e}")
                    results[code] = {'success': False, 'message': str(e)}
                    self.failed_count += 1
                    self.failed_stocks.append(code)

                # 进度报告
                current_time = time.time()
                if completed_count % progress_interval == 0 or current_time - last_progress_time >= 30:
                    elapsed = current_time - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0
                    eta = (self.total_stocks - completed_count) / speed if speed > 0 else 0

                    logger.info(
                        f"进度: {completed_count}/{self.total_stocks} "
                        f"({completed_count / self.total_stocks * 100:.1f}%) | "
                        f"成功: {self.success_count} | "
                        f"跳过: {self.skipped_count} | "
                        f"失败: {self.failed_count} | "
                        f"速度: {speed:.2f} 只/秒 | "
                        f"预计剩余: {eta / 60:.1f} 分钟"
                    )
                    last_progress_time = current_time

        return results

    def print_summary(self, elapsed_time: float):
        """打印同步摘要"""
        print(f"\n{'=' * 70}")
        print(f"同步完成")
        print(f"{'=' * 70}")
        print(f"总股票数: {self.total_stocks}")
        print(f"[OK] 成功: {self.success_count}")
        print(f"[SKIP] 跳过（已有数据）: {self.skipped_count}")
        print(f"[FAIL] 失败: {self.failed_count}")
        print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.1f} 分钟)")

        if self.total_stocks > 0:
            avg_time = elapsed_time / self.total_stocks
            print(f"平均耗时: {avg_time:.2f} 秒/只")

        if self.failed_stocks and len(self.failed_stocks) <= 50:
            print(f"\n失败股票列表: {', '.join(self.failed_stocks)}")
        elif self.failed_stocks:
            print(f"\n失败股票数量: {len(self.failed_stocks)} (列表太长不显示)")

        print(f"{'=' * 70}\n")


class AkshareBatchSyncer:
    """
    Akshare 批量同步器（多线程并发获取）

    核心优势：
    - 使用多线程并发获取多只股票数据
    - 内置防封禁策略（随机休眠、User-Agent 轮换）
    - 支持断点续传
    - 无需 Token，免费使用

    使用方式：
        syncer = AkshareBatchSyncer()
        syncer.sync_by_stock_codes(stock_codes=['600519', '000001', ...])

    Akshare 限流说明：
    - 每分钟建议不超过 100-200 次请求
    - 本实现默认 30 次/分钟，保守策略避免封禁
    """

    def __init__(
            self,
            db: DatabaseManager = None,
            force_refresh: bool = False,
            max_workers: int = 3,
            sleep_min: float = 2.0,
            sleep_max: float = 4.0,
            rate_limit_per_minute: int = 30
    ):
        """
        初始化 Akshare 批量同步器

        Args:
            db: 数据库管理器
            force_refresh: 是否强制刷新（忽略本地缓存）
            max_workers: 最大并发数（默认3，避免触发反爬）
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
            rate_limit_per_minute: 每分钟最大请求数（默认30，保守策略）
        """
        self.db = db or get_db()
        self.force_refresh = force_refresh
        self.max_workers = max_workers
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.rate_limit_per_minute = rate_limit_per_minute
        self.fetcher_manager = DataFetcherManager()

        # 速率限制相关（线程安全）
        import threading
        self._rate_limit_lock = threading.Lock()
        self._call_count = 0  # 当前分钟内的调用次数
        self._minute_start: Optional[float] = None  # 当前计数周期开始时间

        # 统计信息
        self.total_stocks = 0
        self.success_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.failed_stocks = []

        logger.info(f"Akshare 批量同步器初始化完成，并发数: {max_workers}，速率限制: {rate_limit_per_minute} 次/分钟")

    def _check_rate_limit(self) -> None:
        """
        检查并执行速率限制（线程安全）

        流控策略：
        1. 检查是否进入新的一分钟
        2. 如果是，重置计数器
        3. 如果当前分钟调用次数超过限制，强制休眠
        """
        with self._rate_limit_lock:
            current_time = time.time()

            # 检查是否需要重置计数器（新的一分钟）
            if self._minute_start is None:
                self._minute_start = current_time
                self._call_count = 0
            elif current_time - self._minute_start >= 60:
                # 已经过了一分钟，重置计数器
                self._minute_start = current_time
                self._call_count = 0
                logger.debug("Akshare 速率限制计数器已重置")

            # 检查是否超过配额
            if self._call_count >= self.rate_limit_per_minute:
                # 计算需要等待的时间（到下一分钟）
                elapsed = current_time - self._minute_start
                sleep_time = max(0, 60 - elapsed) + 1  # +1 秒缓冲

                logger.warning(
                    f"Akshare 达到速率限制 ({self._call_count}/{self.rate_limit_per_minute} 次/分钟)，"
                    f"等待 {sleep_time:.1f} 秒..."
                )

                time.sleep(sleep_time)

                # 重置计数器
                self._minute_start = time.time()
                self._call_count = 0

            # 增加调用计数
            self._call_count += 1
            logger.debug(f"Akshare 当前分钟调用次数: {self._call_count}/{self.rate_limit_per_minute}")

    def fetch_and_save_stock_data(
            self,
            code: str,
            start_date: str,
            end_date: str
    ) -> tuple:
        """
        获取并保存单只股票数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Tuple[是否成功, 错误信息]
        """
        try:
            # 断点续传检查
            today = date.today()
            if not self.force_refresh and self.db.has_today_data(code, today):
                logger.debug(f"[{code}] 今日数据已存在，跳过获取（断点续传）")
                return True, "skipped"

            # 速率限制检查
            self._check_rate_limit()

            # 随机休眠（防封禁）
            import random
            sleep_time = random.uniform(self.sleep_min, self.sleep_max)
            logger.debug(f"[{code}] 随机休眠 {sleep_time:.2f} 秒...")
            time.sleep(sleep_time)

            # 从数据源获取数据
            logger.debug(f"[{code}] 开始从数据源获取数据...")
            df, source_name = self.fetcher_manager.get_daily_data(
                stock_code=code,
                start_date=start_date,
                end_date=end_date,
                days=90
            )

            if df is None or df.empty:
                return False, "获取数据为空"

            # 保存到数据库
            saved_count = self.db.save_daily_data(df, code, source_name)
            logger.info(f"[{code}] 数据保存成功（来源: {source_name}，新增/更新 {saved_count} 条）")

            return True, "success"

        except Exception as e:
            error_msg = f"获取/保存数据失败: {str(e)}"
            logger.error(f"[{code}] {error_msg}")
            return False, error_msg

    def sync_by_stock_codes(
            self,
            stock_codes: List[str],
            start_date: str = None,
            end_date: str = None,
            progress_interval: int = 10
    ) -> dict:
        """
        按股票代码批量同步数据（多线程并发）

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            progress_interval: 进度报告间隔

        Returns:
            同步统计信息
        """
        if not stock_codes:
            logger.warning("股票代码列表为空")
            return {}

        # 默认日期范围
        if end_date is None:
            end_date = date.today().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = '2025-09-01'

        self.total_stocks = len(stock_codes)
        logger.info(f"开始同步 {self.total_stocks} 只股票数据（Akshare 多线程模式）")
        logger.info(f"日期范围: {start_date} ~ {end_date}")
        logger.info(f"并发数: {self.max_workers}, 休眠范围: {self.sleep_min}-{self.sleep_max} 秒")

        start_time = time.time()
        last_progress_time = start_time

        # 并发获取数据
        results = {}
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(
                    self.fetch_and_save_stock_data,
                    code,
                    start_date,
                    end_date
                ): code
                for code in stock_codes
            }

            # 收集结果
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                completed_count += 1

                try:
                    success, message = future.result()
                    results[code] = {'success': success, 'message': message}

                    if success:
                        if message == "skipped":
                            self.skipped_count += 1
                        else:
                            self.success_count += 1
                    else:
                        self.failed_count += 1
                        self.failed_stocks.append(code)

                except Exception as e:
                    logger.error(f"[{code}] 执行异常: {e}")
                    results[code] = {'success': False, 'message': str(e)}
                    self.failed_count += 1
                    self.failed_stocks.append(code)

                # 进度报告
                current_time = time.time()
                if completed_count % progress_interval == 0 or current_time - last_progress_time >= 30:
                    elapsed = current_time - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0
                    eta = (self.total_stocks - completed_count) / speed if speed > 0 else 0

                    logger.info(
                        f"进度: {completed_count}/{self.total_stocks} "
                        f"({completed_count / self.total_stocks * 100:.1f}%) | "
                        f"成功: {self.success_count} | "
                        f"跳过: {self.skipped_count} | "
                        f"失败: {self.failed_count} | "
                        f"速度: {speed:.2f} 只/秒 | "
                        f"预计剩余: {eta / 60:.1f} 分钟"
                    )
                    last_progress_time = current_time

        return results

    def print_summary(self, elapsed_time: float):
        """打印同步摘要"""
        print(f"\n{'=' * 70}")
        print(f"Akshare 批量同步完成")
        print(f"{'=' * 70}")
        print(f"总股票数: {self.total_stocks}")
        print(f"[OK] 成功: {self.success_count}")
        print(f"[SKIP] 跳过（已有数据）: {self.skipped_count}")
        print(f"[FAIL] 失败: {self.failed_count}")
        print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.1f} 分钟)")

        if self.total_stocks > 0:
            avg_time = elapsed_time / self.total_stocks
            print(f"平均耗时: {avg_time:.2f} 秒/只")

        if self.failed_stocks and len(self.failed_stocks) <= 50:
            print(f"\n失败股票列表: {', '.join(self.failed_stocks)}")
        elif self.failed_stocks:
            print(f"\n失败股票数量: {len(self.failed_stocks)} (列表太长不显示)")

        print(f"{'=' * 70}\n")


class TushareBatchSyncer:
    """
    Tushare 批量同步器（按交易日高效获取）

    核心优势：
    - 按交易日获取数据，而非按股票代码循环
    - 5000 只股票只需 ~220 次请求（每年交易日数）
    - 效率比按股票循环提升 20+ 倍
    - 降低 API 频率限制触发概率

    使用方式：
        syncer = TushareBatchSyncer()
        syncer.sync_by_trade_date(start_date='2025-01-01', end_date='2025-01-31')

    Tushare 配额说明（免费用户）：
    - 每分钟最多 80-120 次请求（根据积分不同）
    - 本实现默认 50 次/分钟，保守策略避免触发限制
    """

    def __init__(
            self,
            db: DatabaseManager = None,
            force_refresh: bool = False,
            retry_times: int = 3,
            retry_delay: float = 1.0,
            rate_limit_per_minute: int = 50
    ):
        """
        初始化 Tushare 批量同步器

        Args:
            db: 数据库管理器
            force_refresh: 是否强制刷新（忽略本地缓存）
            retry_times: 失败重试次数
            retry_delay: 重试延迟（秒）
            rate_limit_per_minute: 每分钟最大请求数（默认50，保守策略）
        """
        self.db = db or get_db()
        self.force_refresh = force_refresh
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.rate_limit_per_minute = rate_limit_per_minute
        self._api: Optional[object] = None
        self._init_api()

        # 速率限制相关
        self._call_count = 0  # 当前分钟内的调用次数
        self._minute_start: Optional[float] = None  # 当前计数周期开始时间

        # 统计信息
        self.total_dates = 0
        self.success_dates = 0
        self.failed_dates = 0
        self.total_records = 0
        self.failed_trade_dates = []

        logger.info(f"Tushare 批量同步器初始化完成，速率限制: {rate_limit_per_minute} 次/分钟")

    def _init_api(self) -> None:
        """初始化 Tushare API"""
        config = get_config()

        if not config.tushare_token:
            logger.error("Tushare Token 未配置，无法使用批量同步模式")
            logger.error("请在 .env 文件中设置 TUSHARE_TOKEN")
            return

        try:
            import tushare as ts
            ts.set_token(config.tushare_token)
            self._api = ts.pro_api()
            logger.info("Tushare API 初始化成功")
        except Exception as e:
            logger.error(f"Tushare API 初始化失败: {e}")
            self._api = None

    def is_available(self) -> bool:
        """检查 Tushare API 是否可用"""
        return self._api is not None

    def _check_rate_limit(self) -> None:
        """
        检查并执行速率限制

        流控策略：
        1. 检查是否进入新的一分钟
        2. 如果是，重置计数器
        3. 如果当前分钟调用次数超过限制，强制休眠
        """
        current_time = time.time()

        # 检查是否需要重置计数器（新的一分钟）
        if self._minute_start is None:
            self._minute_start = current_time
            self._call_count = 0
        elif current_time - self._minute_start >= 60:
            # 已经过了一分钟，重置计数器
            self._minute_start = current_time
            self._call_count = 0
            logger.debug("速率限制计数器已重置")

        # 检查是否超过配额
        if self._call_count >= self.rate_limit_per_minute:
            # 计算需要等待的时间（到下一分钟）
            elapsed = current_time - self._minute_start
            sleep_time = max(0, 60 - elapsed) + 1  # +1 秒缓冲

            logger.warning(
                f"Tushare 达到速率限制 ({self._call_count}/{self.rate_limit_per_minute} 次/分钟)，"
                f"等待 {sleep_time:.1f} 秒..."
            )

            time.sleep(sleep_time)

            # 重置计数器
            self._minute_start = time.time()
            self._call_count = 0

        # 增加调用计数
        self._call_count += 1
        logger.debug(f"Tushare 当前分钟调用次数: {self._call_count}/{self.rate_limit_per_minute}")

    def get_trade_dates(
            self,
            start_date: str,
            end_date: str
    ) -> List[str]:
        """
        获取交易日历

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            交易日列表 (YYYYMMDD 格式)
        """
        if self._api is None:
            logger.error("Tushare API 未初始化")
            return []

        try:
            # 速率限制检查
            self._check_rate_limit()

            # 转换日期格式
            ts_start = start_date.replace('-', '')
            ts_end = end_date.replace('-', '')

            logger.info(f"获取交易日历: {start_date} ~ {end_date}")

            # 调用 trade_cal 接口
            df = self._api.trade_cal(
                exchange='SSE',  # 上交所
                is_open='1',     # 1=开盘
                start_date=ts_start,
                end_date=ts_end,
                fields='cal_date'
            )

            if df is not None and not df.empty:
                trade_dates = df['cal_date'].tolist()
                logger.info(f"获取交易日历成功: {len(trade_dates)} 个交易日")
                return trade_dates
            else:
                logger.warning("交易日历为空")
                return []

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []

    def fetch_daily_by_date(self, trade_date: str) -> Optional[pd.DataFrame]:
        """
        获取指定交易日所有股票数据

        Args:
            trade_date: 交易日 (YYYYMMDD 格式)

        Returns:
            DataFrame 包含当日所有股票数据，失败返回 None
        """
        if self._api is None:
            return None

        for attempt in range(self.retry_times):
            try:
                # 速率限制检查（每次重试前都检查）
                self._check_rate_limit()

                logger.debug(f"获取 {trade_date} 数据 (尝试 {attempt + 1}/{self.retry_times})")

                # 调用 daily 接口获取全市场数据
                df = self._api.daily(trade_date=trade_date)

                if df is not None and not df.empty:
                    logger.debug(f"获取 {trade_date} 数据成功: {len(df)} 只股票")
                    return df
                else:
                    logger.warning(f"获取 {trade_date} 数据为空")
                    return None

            except Exception as e:
                error_msg = str(e).lower()

                # 检测配额超限
                if any(keyword in error_msg for keyword in ['quota', '配额', 'limit', '权限']):
                    logger.error(f"Tushare 配额超限: {e}")
                    logger.error("请等待配额恢复或升级 Tushare 账户")
                    return None

                if attempt < self.retry_times - 1:
                    logger.warning(f"获取 {trade_date} 失败: {e}，{self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"获取 {trade_date} 失败，已达最大重试次数: {e}")
                    return None

        return None

    def normalize_and_save(self, df: pd.DataFrame, trade_date: str) -> int:
        """
        标准化数据并批量保存到数据库

        Args:
            df: Tushare 原始数据
            trade_date: 交易日 (YYYYMMDD)

        Returns:
            保存的记录数
        """
        if df is None or df.empty:
            return 0

        df = df.copy()

        # 列名映射
        column_mapping = {
            'trade_date': 'date',
            'vol': 'volume',
            'ts_code': 'code'
        }

        df = df.rename(columns=column_mapping)

        # 转换日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        # 转换股票代码格式（600519.SH -> 600519）
        if 'code' in df.columns:
            df['code'] = df['code'].apply(lambda x: x.split('.')[0] if '.' in str(x) else x)

        # 成交量单位转换（手 -> 股）
        if 'volume' in df.columns:
            df['volume'] = df['volume'] * 100

        # 成交额单位转换（千元 -> 元）
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 1000

        # 按股票分组保存
        saved_count = 0
        grouped = df.groupby('code')

        for code, group_df in grouped:
            try:
                # 断点续传检查
                if not self.force_refresh:
                    today_dt = datetime.strptime(trade_date, '%Y%m%d').date()
                    if self.db.has_today_data(code, today_dt):
                        continue

                # 保存单只股票数据
                count = self.db.save_daily_data(group_df, code, 'Tushare')
                saved_count += count

            except Exception as e:
                logger.warning(f"保存股票 {code} 数据失败: {e}")

        return saved_count

    def sync_by_trade_date(
            self,
            start_date: str,
            end_date: str,
            progress_interval: int = 5
    ) -> dict:
        """
        按交易日批量同步数据

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            progress_interval: 进度报告间隔

        Returns:
            同步统计信息
        """
        if not self.is_available():
            logger.error("Tushare API 不可用，无法执行批量同步")
            return {}

        # 获取交易日历
        trade_dates = self.get_trade_dates(start_date, end_date)

        if not trade_dates:
            logger.error("未获取到交易日历")
            return {}

        self.total_dates = len(trade_dates)
        logger.info(f"开始批量同步 {self.total_dates} 个交易日的数据")

        start_time = time.time()
        last_progress_time = start_time

        # 按交易日循环获取数据
        for idx, trade_date in enumerate(trade_dates, 1):
            try:
                # 获取当日数据
                df = self.fetch_daily_by_date(trade_date)

                if df is not None:
                    # 标准化并保存
                    saved_count = self.normalize_and_save(df, trade_date)
                    self.total_records += saved_count
                    self.success_dates += 1

                    logger.info(
                        f"[{idx}/{self.total_dates}] {trade_date} | "
                        f"股票数: {len(df)} | "
                        f"保存: {saved_count} 条"
                    )
                else:
                    self.failed_dates += 1
                    self.failed_trade_dates.append(trade_date)
                    logger.error(f"[{idx}/{self.total_dates}] {trade_date} | 获取失败")

                # 进度报告
                current_time = time.time()
                if idx % progress_interval == 0 or current_time - last_progress_time >= 30:
                    elapsed = current_time - start_time
                    speed = idx / elapsed if elapsed > 0 else 0
                    eta = (self.total_dates - idx) / speed if speed > 0 else 0

                    logger.info(
                        f"进度: {idx}/{self.total_dates} ({idx / self.total_dates * 100:.1f}%) | "
                        f"成功: {self.success_dates} | "
                        f"失败: {self.failed_dates} | "
                        f"已保存: {self.total_records} 条 | "
                        f"速度: {speed:.2f} 日/秒 | "
                        f"预计剩余: {eta / 60:.1f} 分钟"
                    )
                    last_progress_time = current_time

            except Exception as e:
                self.failed_dates += 1
                self.failed_trade_dates.append(trade_date)
                logger.error(f"[{idx}/{self.total_dates}] {trade_date} | 处理异常: {e}")

        return {
            'total_dates': self.total_dates,
            'success_dates': self.success_dates,
            'failed_dates': self.failed_dates,
            'total_records': self.total_records,
            'failed_trade_dates': self.failed_trade_dates
        }

    def print_summary(self, elapsed_time: float):
        """打印同步摘要"""
        print(f"\n{'=' * 70}")
        print(f"Tushare 批量同步完成")
        print(f"{'=' * 70}")
        print(f"总交易日数: {self.total_dates}")
        print(f"[OK] 成功: {self.success_dates}")
        print(f"[FAIL] 失败: {self.failed_dates}")
        print(f"总保存记录: {self.total_records} 条")
        print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.1f} 分钟)")

        if self.total_dates > 0:
            avg_time = elapsed_time / self.total_dates
            avg_records = self.total_records / self.total_dates
            print(f"平均耗时: {avg_time:.2f} 秒/日")
            print(f"平均记录: {avg_records:.0f} 条/日")

        if self.failed_trade_dates:
            print(f"\n失败交易日: {', '.join(self.failed_trade_dates[:20])}")
            if len(self.failed_trade_dates) > 20:
                print(f"  ... 还有 {len(self.failed_trade_dates) - 20} 个")

        print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='同步整个A股市场所有股票数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2025-12-01',
        help='开始日期 (YYYY-MM-DD)，默认: 2025-12-01'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help=f'结束日期 (YYYY-MM-DD)，默认: 今天 ({date.today()})'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='并发线程数，默认: 5 (建议3-10)'
    )

    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='强制刷新（忽略本地缓存，重新获取所有数据）'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='断点续传（跳过已有今日数据的股票）'
    )

    parser.add_argument(
        '--market',
        type=str,
        default=None,
        help='过滤市场（如：主板、创业板、科创板）'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制同步数量（测试用）'
    )

    parser.add_argument(
        '--list-only',
        action='store_true',
        help='只获取股票列表，不进行数据同步'
    )

    parser.add_argument(
        '--save-list',
        type=str,
        default=None,
        help='保存股票列表到文件 (CSV格式)'
    )

    parser.add_argument(
        '--by-date',
        action='store_true',
        help='使用 Tushare 按交易日批量获取（推荐，效率提升 20+ 倍）'
    )

    parser.add_argument(
        '--use-akshare',
        action='store_true',
        help='使用 Akshare 多线程批量获取（免费，无需 Token）'
    )

    parser.add_argument(
        '--akshare-workers',
        type=int,
        default=3,
        help='Akshare 并发数（默认: 3，建议 2-5）'
    )

    args = parser.parse_args()

    # 获取股票列表
    df = get_all_a_stocks_from_akshare()

    if df.empty:
        logger.error("获取股票列表失败，退出")
        return 1

    # 保存股票列表
    if args.save_list:
        filename = args.save_list
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"股票列表已保存到: {filename}")

    # 只显示列表
    if args.list_only:
        print(f"\n{'=' * 70}")
        print(f"A股市场股票列表")
        print(f"{'=' * 70}")
        print(f"总数量: {len(df)}")
        print(f"\n按市场分布:")
        for market, count in df['market'].value_counts().items():
            print(f"  - {market}: {count} 只")
        print(f"\n前20只股票:")
        print(df.head(20).to_string(index=False))
        print(f"{'=' * 70}\n")
        return 0

    # ===== Tushare 按交易日批量获取模式 =====
    if args.by_date:
        logger.info("=" * 70)
        logger.info("使用 Tushare 按交易日批量获取模式（高效）")
        logger.info("=" * 70)

        # 创建 Tushare 批量同步器
        batch_syncer = TushareBatchSyncer(
            force_refresh=args.force_refresh
        )

        # 检查 Tushare API 是否可用
        if not batch_syncer.is_available():
            logger.error("Tushare API 不可用，无法使用批量获取模式")
            logger.error("请检查 .env 文件中的 TUSHARE_TOKEN 配置")
            return 1

        # 设置日期范围
        end_date = args.end if args.end else date.today().strftime('%Y-%m-%d')
        start_date = args.start

        logger.info(f"日期范围: {start_date} ~ {end_date}")

        # 执行批量同步
        start_time = time.time()
        results = batch_syncer.sync_by_trade_date(
            start_date=start_date,
            end_date=end_date,
            progress_interval=5
        )
        elapsed_time = time.time() - start_time

        # 打印摘要
        batch_syncer.print_summary(elapsed_time)

        return 0 if batch_syncer.failed_dates == 0 else 1

    # ===== Akshare 多线程批量获取模式 =====
    if args.use_akshare:
        logger.info("=" * 70)
        logger.info("使用 Akshare 多线程批量获取模式（免费，无需 Token）")
        logger.info("=" * 70)

        # 过滤股票列表
        stock_codes = filter_stocks(df, market_filter=args.market, limit=args.limit)

        if not stock_codes:
            logger.error("股票列表为空")
            return 1

        # 创建 Akshare 批量同步器
        akshare_syncer = AkshareBatchSyncer(
            max_workers=args.akshare_workers,
            force_refresh=args.force_refresh
        )

        # 设置日期范围
        end_date = args.end if args.end else date.today().strftime('%Y-%m-%d')
        start_date = args.start

        # 执行批量同步
        start_time = time.time()
        results = akshare_syncer.sync_by_stock_codes(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            progress_interval=10
        )
        elapsed_time = time.time() - start_time

        # 打印摘要
        akshare_syncer.print_summary(elapsed_time)

        return 0 if akshare_syncer.failed_count == 0 else 1

    # ===== 原有模式：按股票代码获取 =====
    # 过滤股票列表
    stock_codes = filter_stocks(df, market_filter=args.market, limit=args.limit)

    if not stock_codes:
        logger.error("股票列表为空")
        return 1

    # 创建同步器
    syncer = StockDataSyncer(
        max_workers=args.workers,
        force_refresh=args.force_refresh
    )

    # 执行同步
    start_time = time.time()
    results = syncer.sync_stocks(
        stock_codes=stock_codes,
        start_date=args.start,
        end_date=args.end,
        progress_interval=50
    )
    elapsed_time = time.time() - start_time

    # 打印摘要
    syncer.print_summary(elapsed_time)

    return 0 if syncer.failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
