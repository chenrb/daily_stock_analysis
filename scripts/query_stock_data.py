#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询数据库中的股票日线数据

用法:
    python scripts/query_stock_data.py                    # 打印到控制台
    python scripts/query_stock_data.py --export csv       # 导出为 CSV
    python scripts/query_stock_data.py --export excel     # 导出为 Excel
    python scripts/query_stock_data.py --code 600519      # 查询指定股票
    python scripts/query_stock_data.py --start 2025-01-01 # 自定义开始日期
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, and_
from src.storage import DatabaseManager, StockDaily
from src.config import setup_env

# 初始化环境
setup_env()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
)
logger = logging.getLogger(__name__)


def query_stock_data(
    start_date: str,
    end_date: str,
    stock_code: str = None,
    db: DatabaseManager = None
):
    """
    查询股票日线数据

    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        stock_code: 股票代码（可选，不指定则查询所有股票）
        db: 数据库管理器

    Returns:
        List[StockDaily]: 查询结果列表
    """
    if db is None:
        db = DatabaseManager.get_instance()

    # 解析日期
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()

    logger.info(f"查询日期范围: {start} 至 {end}")
    if stock_code:
        logger.info(f"股票代码: {stock_code}")

    # 构建查询
    with db.get_session() as session:
        conditions = [
            StockDaily.date >= start,
            StockDaily.date <= end,
        ]

        if stock_code:
            conditions.append(StockDaily.code == stock_code)

        results = session.execute(
            select(StockDaily)
            .where(and_(*conditions))
            .order_by(StockDaily.code, StockDaily.date)
        ).scalars().all()

    logger.info(f"查询到 {len(results)} 条记录")
    return list(results)


def print_results(results):
    """打印查询结果到控制台"""
    if not results:
        print("没有找到数据")
        return

    # 按股票分组
    from collections import defaultdict
    stocks = defaultdict(list)
    for r in results:
        stocks[r.code].append(r)

    print(f"\n{'='*100}")
    print(f"查询结果汇总")
    print(f"{'='*100}")
    print(f"股票数量: {len(stocks)}")
    print(f"总记录数: {len(results)}")
    print(f"{'='*100}\n")

    for code, records in sorted(stocks.items()):
        name = records[0].code  # 可以扩展获取股票名称
        print(f"\n【{code}】共 {len(records)} 条记录")
        print(f"{'日期':<12} {'开盘':<8} {'最高':<8} {'最低':<8} {'收盘':<8} {'成交量':<12} {'涨跌幅':<8}")
        print("-" * 80)

        for r in records[-10:]:  # 只显示最近10条
            volume_str = f"{r.volume/10000:.1f}万" if r.volume else "N/A"
            pct_str = f"{r.pct_chg:.2f}%" if r.pct_chg is not None else "N/A"
            print(f"{str(r.date):<12} {r.open:<8.2f} {r.high:<8.2f} {r.low:<8.2f} "
                  f"{r.close:<8.2f} {volume_str:<12} {pct_str:<8}")

        if len(records) > 10:
            print(f"... (还有 {len(records) - 10} 条历史记录)")


def export_to_csv(results, output_file: str = None):
    """导出为 CSV"""
    import pandas as pd

    if not results:
        print("没有数据可导出")
        return

    # 转换为 DataFrame
    data = [r.to_dict() for r in results]
    df = pd.DataFrame(data)

    # 选择和排序列
    columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume',
               'amount', 'pct_chg', 'ma5', 'ma10', 'ma20', 'volume_ratio', 'data_source']
    df = df[[col for col in columns if col in df.columns]]

    # 默认文件名
    if output_file is None:
        output_file = f"stock_data_{date.today().strftime('%Y%m%d')}.csv"

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已导出到: {output_file}")
    print(f"总计 {len(results)} 条记录，{len(df['code'].unique())} 只股票")


def export_to_excel(results, output_file: str = None):
    """导出为 Excel"""
    import pandas as pd

    if not results:
        print("没有数据可导出")
        return

    # 转换为 DataFrame
    data = [r.to_dict() for r in results]
    df = pd.DataFrame(data)

    # 选择和排序列
    columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume',
               'amount', 'pct_chg', 'ma5', 'ma10', 'ma20', 'volume_ratio', 'data_source']
    df = df[[col for col in columns if col in df.columns]]

    # 默认文件名
    if output_file is None:
        output_file = f"stock_data_{date.today().strftime('%Y%m%d')}.xlsx"

    # 按股票分多个 Sheet
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for code in sorted(df['code'].unique()):
            df_code = df[df['code'] == code].sort_values('date')
            # Excel sheet 名称最多31个字符
            sheet_name = f"{code}"[:31]
            df_code.to_excel(writer, sheet_name=sheet_name, index=False)

        # 汇总表
        df_summary = df.groupby('code').agg({
            'date': ['min', 'max', 'count'],
            'close': 'last'
        }).round(2)
        df_summary.columns = ['最早日期', '最新日期', '记录数', '最新收盘价']
        df_summary.to_excel(writer, sheet_name='汇总', index=True)

    print(f"已导出到: {output_file}")
    print(f"总计 {len(results)} 条记录，{len(df['code'].unique())} 只股票")


def main():
    parser = argparse.ArgumentParser(
        description='查询数据库中的股票日线数据',
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
        '--code',
        type=str,
        default=None,
        help='股票代码（如 600519），不指定则查询所有股票'
    )

    parser.add_argument(
        '--export',
        type=str,
        choices=['csv', 'excel', 'print'],
        default='print',
        help='导出方式: csv, excel, print (默认: print)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件名（可选）'
    )

    args = parser.parse_args()

    # 默认结束日期为今天
    end_date = args.end or date.today().strftime('%Y-%m-%d')

    # 查询数据
    results = query_stock_data(
        start_date=args.start,
        end_date=end_date,
        stock_code=args.code
    )

    if not results:
        print("没有找到符合条件的数据")
        return 0

    # 输出结果
    if args.export == 'csv':
        export_to_csv(results, args.output)
    elif args.export == 'excel':
        export_to_excel(results, args.output)
    else:
        print_results(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
