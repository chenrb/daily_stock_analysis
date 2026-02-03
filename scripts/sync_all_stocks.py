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
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
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
from sqlalchemy import select, and_, func
from src.storage import StockDaily

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
            start_date = '2025-12-01'

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
                        f"({completed_count/self.total_stocks*100:.1f}%) | "
                        f"成功: {self.success_count} | "
                        f"跳过: {self.skipped_count} | "
                        f"失败: {self.failed_count} | "
                        f"速度: {speed:.2f} 只/秒 | "
                        f"预计剩余: {eta/60:.1f} 分钟"
                    )
                    last_progress_time = current_time

        return results

    def print_summary(self, elapsed_time: float):
        """打印同步摘要"""
        print(f"\n{'='*70}")
        print(f"同步完成")
        print(f"{'='*70}")
        print(f"总股票数: {self.total_stocks}")
        print(f"[OK] 成功: {self.success_count}")
        print(f"[SKIP] 跳过（已有数据）: {self.skipped_count}")
        print(f"[FAIL] 失败: {self.failed_count}")
        print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")

        if self.total_stocks > 0:
            avg_time = elapsed_time / self.total_stocks
            print(f"平均耗时: {avg_time:.2f} 秒/只")

        if self.failed_stocks and len(self.failed_stocks) <= 50:
            print(f"\n失败股票列表: {', '.join(self.failed_stocks)}")
        elif self.failed_stocks:
            print(f"\n失败股票数量: {len(self.failed_stocks)} (列表太长不显示)")

        print(f"{'='*70}\n")


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
        print(f"\n{'='*70}")
        print(f"A股市场股票列表")
        print(f"{'='*70}")
        print(f"总数量: {len(df)}")
        print(f"\n按市场分布:")
        for market, count in df['market'].value_counts().items():
            print(f"  - {market}: {count} 只")
        print(f"\n前20只股票:")
        print(df.head(20).to_string(index=False))
        print(f"{'='*70}\n")
        return 0

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
