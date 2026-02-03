from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from tqdm import tqdm

import akshare as ak
import numpy as np
import pandas as pd

# 调整后的参数 - 2个月窗口期策略
SHORT_WIN = 10      # 短期：约2周
MID_WIN   = 20      # 中期：约1个月
LONG_WIN  = 40      # 长期：约2个月
NEW_HIGH_MAX_BD = 3        # 几日之内创 40 日新高
BREAK_THRE = 0.90          # 回踩不破 90%（轻微回调后重新突破）
CONSOL_VOLAT = 0.20        # 10 日振幅上限（缩短观察期）
RPS_THRESHOLD = 85         # RPS阈值
MAX_WORKERS = 10           # 最大线程数

def get_all_a_share_codes():
    """批量获取全部 A 股代码、名称、行业（使用实时行情接口）"""
    try:
        print("   使用实时行情接口批量获取股票信息...")
        spot_df = ak.stock_zh_a_spot_em()

        # 提取需要的列：代码、名称、行业等
        result = []
        for _, row in spot_df.iterrows():
            code = row['代码']
            name = row['名称']
            industry = row.get('行业', '未分类')

            # 过滤掉非A股（如北交所等）
            if not code.startswith(('00', '30', '60', '68')):
                continue

            result.append({
                'code': code,
                'name': name,
                'industry': industry if pd.notna(industry) else '未分类',
                'list_date': pd.Timestamp('2000-01-01')
            })

        df = pd.DataFrame(result)
        print(f"   成功获取 {len(df)} 只股票信息")
        return df

    except Exception as e:
        print(f"   批量接口失败: {e}，使用简化模式...")
        stock_info = ak.stock_info_a_code_name()
        result = []
        for _, row in stock_info.iterrows():
            code = row['code']
            if code.startswith(('00', '30', '60', '68')):
                result.append({
                    'code': code,
                    'name': row['name'],
                    'industry': '未分类',
                    'list_date': pd.Timestamp('2000-01-01')
                })
        return pd.DataFrame(result)


def calc_return(code, win):
    """计算单只股票指定期间的涨幅"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                adjust="hfq",
                                start_date=(datetime.today()-timedelta(days=win+30)).strftime('%Y%m%d'))
        if len(df) < win:
            return np.nan
        return (df.iloc[-1]['收盘'] - df.iloc[-win]['收盘']) / df.iloc[-win]['收盘']
    except Exception:
        return np.nan


def calc_return_batch(codes_win):
    """批量计算涨幅（用于多线程）"""
    code, win = codes_win
    return code, calc_return(code, win)


def rps_single_mt(codes, win, desc="", max_workers=MAX_WORKERS):
    """多线程计算 RPS"""
    print(f"   计算{desc}RPS ({win}日窗口) - 使用{max_workers}线程...")

    ret = {}
    tasks = [(code, win) for code in codes]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calc_return_batch, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"   {desc}", ncols=80):
            try:
                code, value = future.result()
                if not np.isnan(value):
                    ret[code] = value
            except Exception:
                pass

    ret = pd.Series(ret)
    if len(ret) == 0:
        return ret
    rank = ret.rank(method='min', ascending=False)
    rps = (rank-1)/(len(rank)-1)*100
    return rps


def rps_by_industry(stock_df: pd.DataFrame, win, desc="", max_workers=MAX_WORKERS) -> pd.Series:
    """按行业分组计算 RPS（多线程版本）"""
    all_rps = {}
    for industry, group in stock_df.groupby('industry'):
        codes = group['code'].values
        if len(codes) < 5:
            continue
        print(f"   行业 [{industry}] - {len(codes)}只股票")
        industry_rps = rps_single_mt(codes, win, f"{industry} ", max_workers)
        all_rps.update(industry_rps.to_dict())
    return pd.Series(all_rps)


def check_breakout(code):
    """检查单只股票是否突破（用于多线程）"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                adjust="hfq",
                                start_date=(datetime.today()-timedelta(days=LONG_WIN*2)).strftime('%Y%m%d'))
        if len(df) < LONG_WIN:
            return False
        latest_close = df.iloc[-1]['收盘']
        hh40 = df['最高'].rolling(LONG_WIN).max()
        new_high_days = (hh40 == hh40.iloc[-1]).iloc[-NEW_HIGH_MAX_BD-1:].sum()
        if new_high_days == 0:
            return False
        price_ratio = latest_close / hh40.iloc[-1]
        if price_ratio < BREAK_THRE or price_ratio > 1.02:
            return False
        low10  = df['最低'].iloc[-10:].min()
        high10 = df['最高'].iloc[-10:].max()
        if (high10/low10 - 1) > CONSOL_VOLAT:
            return False
        return True
    except Exception:
        return False


def filter_breakouts_mt(codes, max_workers=MAX_WORKERS):
    """多线程过滤突破股"""
    print(f"   突破检测 - 使用{max_workers}线程...")
    final = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_breakout, code): code for code in codes}

        for future in tqdm(as_completed(futures), total=len(futures), desc="   突破检测", ncols=80):
            try:
                if future.result():
                    code = futures[future]
                    final.append(code)
            except Exception:
                pass

    return final


def main():
    print('='*60)
    print('A股RPS行业筛选策略 - 2个月窗口期（多线程优化版）')
    print('='*60)

    print('\n1. 获取股票代码表...')
    list_df = get_all_a_share_codes()
    cutoff = datetime.today() - timedelta(days=LONG_WIN*2)
    before_count = len(list_df)
    list_df = list_df[list_df['list_date'] <= cutoff]
    after_count = len(list_df)
    if before_count > after_count:
        print(f'   剔除上市不足{LONG_WIN*2}日的新股: {before_count} -> {after_count} 只')

    print(f'\n2. 按行业计算三段 RPS ({SHORT_WIN}/{MID_WIN}/{LONG_WIN}日)...')
    print(f'   覆盖 {list_df["industry"].nunique()} 个行业，{len(list_df)} 只股票')

    # 分别计算三段RPS
    rps10 = rps_by_industry(list_df, SHORT_WIN, "10日")
    rps20 = rps_by_industry(list_df, MID_WIN, "20日")
    rps40 = rps_by_industry(list_df, LONG_WIN, "40日")

    # 合并结果
    rps_df = pd.DataFrame({'rps10':rps10, 'rps20':rps20, 'rps40':rps40})
    rps_df = rps_df.dropna()

    # 条件 A：三段≥RPS_THRESHOLD
    cond_rps = (rps_df>=RPS_THRESHOLD).all(axis=1)
    cand = rps_df[cond_rps].index
    print(f'\n   通过 RPS≥{RPS_THRESHOLD} 筛选: {len(cand)} 只')

    if len(cand) == 0:
        print('\n暂无符合条件的股票，程序结束。')
        return

    print('\n3. 近期新高 + 基底突破过滤...')
    final = filter_breakouts_mt(cand, max_workers=MAX_WORKERS)

    print(f'\n4. 最终候选股（{len(final)} 只）：')
    print('='*60)
    if final:
        res = pd.DataFrame({'code':final})
        res = res.merge(list_df, on='code')
        res = res.merge(rps_df, left_on='code', right_index=True)
        # 按行业分组显示
        for industry, group in res.groupby('industry'):
            print(f'\n【{industry}】({len(group)}只)')
            print(group.sort_values(['rps40','rps20'], ascending=False)[
                ['code', 'name', 'rps10', 'rps20', 'rps40']].to_string(index=False))
        # 保存结果
        filename = f'rps_industry_break_{datetime.today().strftime("%Y%m%d")}.csv'
        res.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f'\n已保存至: {filename}')
        print('='*60)
    else:
        print('暂无同时满足"三段强+刚突破"的股票。')


if __name__ == '__main__':
    main()
