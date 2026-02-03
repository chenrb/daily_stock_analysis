#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量同步股票日线数据到数据库

使用项目现有的数据获取逻辑，批量获取并保存股票数据。

用法:
    # 同步配置文件中的所有股票（从2025-12-01至今）
    python scripts/sync_stock_data.py

    # 指定股票列表
    python scripts/sync_stock_data.py --codes 600519,000001,300750

    # 自定义日期范围
    python scripts/sync_stock_data.py --start 2025-01-01 --end 2025-12-31

    # 强制刷新（忽略本地缓存）
    python scripts/sync_stock_data.py --force-refresh

    # 并发同步（默认3个线程）
    python scripts/sync_stock_data.py --workers 5

    # 仅同步，不进行AI分析
    python scripts/sync_stock_data.py --no-analyze
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import setup_env, get_config, Config
setup_env()

from src.storage import DatabaseManager, get_db
from data_provider import DataFetcherManager
from sqlalchemy import select, and_
from src.storage import StockDaily

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
)
logger = logging.getLogger(__name__)


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
    ) -> Tuple[bool, str]:
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
                logger.info(f"[{code}] 今日数据已存在，跳过获取（断点续传）")
                return True, "skipped"

            # 从数据源获取数据
            logger.info(f"[{code}] 开始从数据源获取数据...")
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
        end_date: str = None
    ) -> dict:
        """
        批量同步股票数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

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

        # 并发获取数据
        results = {}
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

        return results

    def print_summary(self):
        """打印同步摘要"""
        print(f"\n{'='*60}")
        print(f"同步完成")
        print(f"{'='*60}")
        print(f"总股票数: {self.total_stocks}")
        print(f"[OK] 成功: {self.success_count}")
        print(f"[SKIP] 跳过（已有数据）: {self.skipped_count}")
        print(f"[FAIL] 失败: {self.failed_count}")

        if self.failed_stocks:
            print(f"\n失败股票列表: {', '.join(self.failed_stocks)}")

        print(f"{'='*60}\n")


def get_stock_list_from_config() -> List[str]:
    """从配置文件获取股票列表"""
    config = get_config()
    stock_list = config.stock_list or []

    # stock_list 可能是列表或字符串
    if isinstance(stock_list, str):
        if not stock_list.strip():
            logger.warning("配置文件中未设置 STOCK_LIST")
            return []
        # 解析股票列表（逗号分隔）
        codes = [code.strip() for code in stock_list.split(',') if code.strip()]
    elif isinstance(stock_list, list):
        codes = stock_list
    else:
        logger.warning("配置文件中 STOCK_LIST 格式不正确")
        return []

    return codes


def get_stock_list_from_db() -> List[str]:
    """从数据库获取已有数据的股票列表"""
    db = get_db()

    with db.get_session() as session:
        # 查询所有不重复的股票代码
        results = session.execute(
            select(StockDaily.code).distinct()
        ).scalars().all()

        return list(results)


def main():
    parser = argparse.ArgumentParser(
        description='批量同步股票日线数据到数据库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--codes',
        type=str,
        default=None,
        help='股票代码列表（逗号分隔），如: 600519,000001,300750'
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
        default=3,
        help='并发线程数，默认: 3'
    )

    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='强制刷新（忽略本地缓存，重新获取所有数据）'
    )

    parser.add_argument(
        '--from-db',
        action='store_true',
        help='从数据库获取已有股票列表（同步数据库中所有股票的最新数据）'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='从配置文件获取所有股票列表'
    )

    args = parser.parse_args()

    # 获取股票列表
    stock_codes = None

    if args.codes:
        # 命令行指定
        stock_codes = [code.strip() for code in args.codes.split(',') if code.strip()]
        logger.info(f"使用命令行指定的股票列表: {stock_codes}")
    elif args.from_db:
        # 从数据库获取
        stock_codes = get_stock_list_from_db()
        logger.info(f"从数据库获取到 {len(stock_codes)} 只股票")
    elif args.all:
        # 从配置文件获取
        stock_codes = get_stock_list_from_config()
        logger.info(f"从配置文件获取到 {len(stock_codes)} 只股票")
    else:
        # 默认从配置文件
        stock_codes = get_stock_list_from_config()
        if not stock_codes:
            logger.error("未找到股票列表，请使用 --codes 指定或配置 STOCK_LIST")
            return 1

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
        end_date=args.end
    )
    elapsed_time = time.time() - start_time

    # 打印摘要
    syncer.print_summary()
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")

    return 0 if syncer.failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
