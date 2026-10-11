"""依据真实 chunk 构造 100 条召回评测集。

设计原则（针对上一版"看答案出题"的偏差修正）：
1. gold id 一律从 chroma_db 实读，不手写、不构造。
2. 库里每份文档被完整灌入 2 遍（同一文本 2 个 id），因此事实型题目
   天然是 multi-gold —— exact 题标全部重复 id，真实反映"标注完整性"要求。
3. 分层比例对齐真实语料分布（货币政策 164 / 白皮书 134 / 投资者情绪 110 /
   全球电商 100 / 贝莱德 72）。5 份文档共 290 条唯一内容。
4. 约 30% harder cases：模糊口语 query、跨文档对比（单文档无法回答）、
   无答案 / 否定题（库外知识或库内不存在的字段）。

输出: evaluate/data/sample.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from agents.vector_search.settings import CHROMA_PERSIST_DIR

OUT_PATH = Path(__file__).resolve().parent / "data" / "sample.jsonl"
COLLECTION = "default"

SRC = {
    "货币": "中国人民银行2025年第四季度中国货币政策执行报告63页.pdf",
    "白皮书": "2026数字人电商直播白皮书.pdf",
    "情绪": "长江商学院2025年投资者情绪问卷调查CKISS四季度报告63页.pdf",
    "电商": "全球电商行业AI应用研究报告2025.pdf",
    "贝莱德": "贝莱德BlackRock2026年全球投资展望报告17页.pdf",
}


def load_index() -> dict[str, dict[int, list[str]]]:
    """返回 {source: {chunk_index: [id, ...]}}（同一文本的重复副本 id 全部保留）。"""
    project_root = Path(__file__).resolve().parents[1]
    persist = CHROMA_PERSIST_DIR
    if not Path(persist).is_absolute():
        persist = str(project_root / persist)
    client = chromadb.PersistentClient(
        path=persist, settings=Settings(anonymized_telemetry=False)
    )
    data = client.get_collection(COLLECTION).get()

    index: dict[str, dict[int, list[str]]] = {}
    for cid, meta in zip(data["ids"], data["metadatas"]):
        meta = meta or {}
        src = str(meta.get("source", ""))
        ci = int(meta.get("chunk_index", -1))
        index.setdefault(src, {}).setdefault(ci, []).append(cid)
    return index


def main() -> None:
    idx_map = load_index()

    def ids(doc: str, *chunk_indexes: int) -> list[str]:
        src = SRC[doc]
        out: list[str] = []
        for ci in chunk_indexes:
            got = idx_map.get(src, {}).get(ci)
            if not got:
                raise SystemExit(f"[FATAL] 未找到 {src} chunk_index={ci}")
            out.extend(got)
        seen: set[str] = set()
        return [x for x in out if not (x in seen or seen.add(x))]

    rows: list[dict] = []

    def add(query: str, gold: list[str], kind: str, doc: str, note: str = "") -> None:
        rows.append({
            "query": query,
            "collection": COLLECTION,
            "gold_chunk_ids": gold,
            "kind": kind,
            "doc": doc,
            "note": note,
        })

    # =========================================================================
    # A. 精确事实题（单文档、答案唯一；gold 含该文本的全部真实副本 id）
    # =========================================================================

    # ---------------- 白皮书 ----------------
    add("2026数字人电商直播白皮书由哪两家机构联合发布", ids("白皮书", 0), "exact_single", "白皮书")
    add("京东推出的新一代数字人大模型叫什么名字", ids("白皮书", 1), "exact_single", "白皮书")
    add("联想与京东数字人合作解决了直播的哪些痛点", ids("白皮书", 4), "exact_single", "白皮书")
    add("数字人电商直播的定义是什么", ids("白皮书", 6), "exact_single", "白皮书")
    add("数字人电商直播行业的发展分为哪几个阶段", ids("白皮书", 8), "exact_single", "白皮书")
    add("数字人电商直播的快速成长期对应哪几年", ids("白皮书", 9), "exact_single", "白皮书")
    add("中国AI数字人2024年的市场增速是多少", ids("白皮书", 10), "exact_single", "白皮书")
    add("电商数字人直播行业的第一梯队包括哪些平台", ids("白皮书", 11), "exact_single", "白皮书")
    add("快手电商在数字人直播上的合规体系有何特点", ids("白皮书", 12), "exact_single", "白皮书")
    add("2026年Q1京东数字人累计服务商家数量突破多少", ids("白皮书", 22), "exact_single", "白皮书")
    add("数字人电商直播能力评估的五维一体框架包含哪些维度", ids("白皮书", 27), "exact_single", "白皮书")

    # ---------------- 货币政策报告 ----------------
    add("2025年中国GDP同比增长多少", ids("货币", 0), "exact_single", "货币")
    add("2025年末社会融资规模存量同比增速是多少", ids("货币", 1), "exact_single", "货币")
    add("2025年末广义货币供应量M2同比增长多少", ids("货币", 1), "exact_single", "货币")
    add("2025年12月新发放企业贷款和个人住房贷款利率分别在什么水平", ids("货币", 1), "exact_single", "货币")
    add("2025年末人民币对美元汇率收盘价是多少", ids("货币", 1), "exact_single", "货币")
    add("2025年末CFETS人民币汇率指数为多少", ids("货币", 1), "exact_single", "货币")
    add("2025年存款准备金率下调了多少个百分点，释放多少长期流动性", ids("货币", 11), "exact_single", "货币")
    add("2025年末金融机构超额存款准备金率是多少", ids("货币", 11), "exact_single", "货币")
    add("2025年末人民币贷款余额和本外币存款余额分别是多少", ids("货币", 12, 13), "exact_single", "货币")
    add("2025年12月1年期和5年期以上LPR分别是多少", ids("货币", 14), "exact_single", "货币")
    add("2025年12月新发放贷款加权平均利率是多少", ids("货币", 15), "exact_single", "货币")
    add("2025年12月企业贷款加权平均利率和个人住房贷款加权平均利率分别是多少", ids("货币", 15), "exact_single", "货币")
    add("2025年12月活期和3个月以内大额美元存款加权平均利率分别是多少", ids("货币", 17), "exact_single", "货币")
    add("2025年末科技贷款和绿色贷款分别同比增长多少", ids("货币", 19), "exact_single", "货币")
    add("2025年末养老产业贷款和数字经济产业贷款分别同比增长多少", ids("货币", 19), "exact_single", "货币")
    add("2025年末直接融资在社会融资规模存量中的占比约多少", ids("货币", 20), "exact_single", "货币")
    add("2025年末M2、M1、M0余额分别是多少", ids("货币", 20), "exact_single", "货币")
    add("2025年社会融资规模存量是多少，全年增量是多少", ids("货币", 21), "exact_single", "货币")
    add("2025年全年政府债券净融资多少，同比多增多少", ids("货币", 21), "exact_single", "货币")
    add("2025年全年企业债券净融资多少", ids("货币", 21), "exact_single", "货币")
    add("2025年末人民币对美元汇率中间价是多少，较上年末升值多少", ids("货币", 23), "exact_single", "货币")
    add("2025年买断式逆回购和中期借贷便利合计净投放约多少万亿元", ids("货币", 25), "exact_single", "货币")
    add("2025年末MLF余额是多少，比年初增加多少", ids("货币", 26), "exact_single", "货币")
    add("2025年累计开展常备借贷便利SLF操作多少亿元", ids("货币", 26), "exact_single", "货币")
    add("2025年末SLF余额是多少", ids("货币", 26), "exact_single", "货币")

    # ---------------- 投资者情绪报告 ----------------
    add("2025年美国标普500全年上涨多少", ids("情绪", 1), "exact_single", "情绪")
    add("2025年日经225和德国DAX全年分别上涨多少", ids("情绪", 1), "exact_single", "情绪")
    add("2025年上证综指全年累计涨幅是多少", ids("情绪", 1), "exact_single", "情绪")
    add("2025年深证综指全年上涨多少", ids("情绪", 1), "exact_single", "情绪")
    add("2025年恒生指数全年上涨多少，第四季度回调多少", ids("情绪", 1), "exact_single", "情绪")
    add("截至2025年第三季度末A股整体估值指标较2024年末上升多少", ids("情绪", 2), "exact_single", "情绪")
    add("2025年全年货物和服务净出口拉动GDP增长多少个百分点", ids("情绪", 2), "exact_single", "情绪")
    add("2025年中国新生婴儿数量是多少，八年内下降了多少", ids("情绪", 3), "exact_single", "情绪")
    add("2025年末全国70个大中城市新建商品住宅价格同比下降多少", ids("情绪", 3), "exact_single", "情绪")
    add("2025年黄金价格全年涨幅达到多少", ids("情绪", 4), "exact_single", "情绪")
    add("CKISS问卷调查的总有效回收样本计划多少份", ids("情绪", 0), "exact_single", "情绪")
    add("2025年12月认为A股会上涨的受访者占比约多少", ids("情绪", 6), "exact_single", "情绪")
    add("2025年12月认为港股会上涨的受访者占比约多少", ids("情绪", 7), "exact_single", "情绪")
    add("2025年12月金融业受访者对港股的看涨比例比9月回调多少个百分点", ids("情绪", 7), "exact_single", "情绪")
    add("2025年12月愿意投资股票的净增加人数占比是多少", ids("情绪", 8), "exact_single", "情绪")
    add("2025年12月愿意投资房地产的净增加人数占比约多少", ids("情绪", 9), "exact_single", "情绪")
    add("2025年12月受访者的预期通胀率约是多少", ids("情绪", 10), "exact_single", "情绪")
    add("2025年12月认为中美关系对未来投资有重大影响的受访者占比约多少", ids("情绪", 11), "exact_single", "情绪")

    # ---------------- 全球电商AI报告 ----------------
    add("2024年全球B2C电商销售额达到多少万亿美元", ids("电商", 2), "exact_single", "电商")
    add("约多少比例的中国人口曾进行过网购", ids("电商", 4), "exact_single", "电商")
    add("2024年美国电子商务市场规模估计为多少，其中B2C交易额多少", ids("电商", 5), "exact_single", "电商")
    add("2024年日本电商市场规模增长多少，交易额达到多少", ids("电商", 7), "exact_single", "电商")
    add("2024年印度电商市场规模达到多少，同比增长多少", ids("电商", 8), "exact_single", "电商")
    add("2024年印度电商交易额中跨境电商占比多少", ids("电商", 9), "exact_single", "电商")
    add("印度电商最畅销的三大品类是什么", ids("电商", 10), "exact_single", "电商")
    add("2024年印度电商最主要的支付方式是什么，占比多少", ids("电商", 10), "exact_single", "电商")
    add("2024年印尼电商交易中移动端和PC端分别占比多少", ids("电商", 12), "exact_single", "电商")
    add("印尼电商市场的四大玩家分别是谁", ids("电商", 13), "exact_single", "电商")
    add("全球有多少比例的购物者从未订阅过电子邮件简报", ids("电商", 15), "exact_single", "电商")
    add("多少比例的购物者因配送费用过高而放弃完成购买", ids("电商", 16), "exact_single", "电商")

    # ---------------- 贝莱德投资展望 ----------------
    add("贝莱德预计截至2030年全球企业在人工智能领域的资本支出目标规模是多少", ids("贝莱德", 4), "exact_single", "贝莱德")
    add("贝莱德认为美国长期维持的经济增长趋势是多少", ids("贝莱德", 4), "exact_single", "贝莱德")
    add("贝莱德测算人工智能投资对应的合理回报率区间是多少", ids("贝莱德", 9), "exact_single", "贝莱德")
    add("彭博共识预测美国超大规模企业2025至2030年年收入增长额是多少", ids("贝莱德", 9), "exact_single", "贝莱德")
    add("贝莱德估算人工智能驱动型新收入到2030年的规模区间是多少", ids("贝莱德", 11), "exact_single", "贝莱德")
    add("截至11月26日云计算支出规模最大企业的平均债务权益比是多少", ids("贝莱德", 12), "exact_single", "贝莱德")
    add("贝莱德预计到2030年人工智能数据中心电力需求占美国当前电力需求的多少", ids("贝莱德", 17), "exact_single", "贝莱德")
    add("LSEG数据显示标普500等权重指数与市值加权指数今年涨幅分别是多少", ids("贝莱德", 14), "exact_single", "贝莱德")

    # =========================================================================
    # B. 多 gold 题（同一事实在库中多处出现 / 同一 chunk 多角度提问）
    # =========================================================================
    add("2025年科技贷款同比增长多少", ids("货币", 19), "multi_gold", "货币", "同事实在摘要与正文重复")
    add("2025年末人民币汇率指数和中间价的变动情况", ids("货币", 1, 23), "multi_gold", "货币", "摘要与正文两处")
    add("2025年是中国什么规划的收官之年", ids("货币", 0, 11), "multi_gold", "货币", "摘要与正文两处")
    add("数字人电商直播的快速成长期是哪几年", ids("白皮书", 8, 9), "multi_gold", "白皮书", "阶段总述与细分表述分属两chunk")
    add("2025年A股上涨主要由什么驱动，而非盈利改善", ids("情绪", 2), "multi_gold", "情绪")
    add("数字人电商直播的价值本质是什么", ids("白皮书", 6), "multi_gold", "白皮书")

    # =========================================================================
    # C. 跨文档对比题（单文档无法回答，必须同时召回两个文档）
    # =========================================================================
    add("对比白皮书与全球电商AI报告，数字人与AI分别如何降低电商运营成本",
        ids("白皮书", 6) + ids("电商", 1), "cross_doc", "多文档", "需同时命中两份文档")
    add("白皮书和货币政策报告分别提到了哪些2026年展望",
        ids("白皮书", 8) + ids("货币", 2), "cross_doc", "多文档")
    add("投资者情绪报告与贝莱德展望对2026年市场风险的判断有何异同",
        ids("情绪", 4) + ids("贝莱德", 3), "cross_doc", "多文档")
    add("全球电商AI报告与投资者情绪报告都提到中国内需，各自结论是什么",
        ids("电商", 4) + ids("情绪", 3), "cross_doc", "多文档")

    # =========================================================================
    # D. 模糊 / 口语化 query（检验语义召回，措辞刻意不与原文重合）
    # =========================================================================
    add("京东搞的那个数字人直播报告是谁一起做的", ids("白皮书", 0), "vague", "白皮书", "口语化，无'白皮书/发布'字样")
    add("钱变便宜了吗，央行去年降了多少利率", ids("货币", 24), "vague", "货币", "口语化，不出现'逆回购'")
    add("现在大家还敢买房子吗", ids("情绪", 8), "vague", "情绪")
    add("做直播太贵了怎么办", ids("白皮书", 6), "vague", "白皮书")
    add("最近金价涨得怎么样", ids("情绪", 4), "vague", "情绪")
    add("美国人现在网购方便了吗", ids("电商", 5), "vague", "电商")
    add("AI烧钱这么猛，钱从哪来", ids("贝莱德", 12), "vague", "贝莱德")
    add("银行里钱多不多", ids("货币", 11), "vague", "货币")
    add("大家对股市还乐观吗", ids("情绪", 6), "vague", "情绪")
    add("印度人上网买东西多吗", ids("电商", 8), "vague", "电商")

    # =========================================================================
    # E. 无答案 / 否定题（库内无对应内容，gold 为空 —— 用于测误召回）
    # =========================================================================
    add("特斯拉2025年全年的交付量是多少", [], "no_answer", "库外", "库中无特斯拉数据")
    add("苹果公司2026年第一财季的净利润是多少", [], "no_answer", "库外")
    add("2026年世界杯的冠军是哪支球队", [], "no_answer", "库外")
    add("这份知识库里有没有关于新能源汽车电池技术的内容", [], "no_answer", "库内不存在")
    add("请列出库中提到的所有上市公司股票代码", [], "no_answer", "库内不存在")
    add("白皮书里京东数字人的日活跃用户数是多少", [], "no_answer", "库内不存在", "提到了商家数但无日活数据")

    # =========================================================================
    from collections import Counter
    kinds = Counter(r["kind"] for r in rows)
    docs = Counter(r["doc"] for r in rows)
    multi = sum(1 for r in rows if len(r["gold_chunk_ids"]) > 1)
    noans = sum(1 for r in rows if not r["gold_chunk_ids"])

    if len(rows) != 100:
        raise SystemExit(f"[FATAL] 期望 100 条，实际 {len(rows)} 条")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"写出 {len(rows)} 条 -> {OUT_PATH}")
    print("题型分布:", dict(kinds))
    print("文档分布:", dict(docs))
    print(f"多 gold 条数: {multi} ({multi/len(rows):.0%})")
    print(f"无答案条数  : {noans} ({noans/len(rows):.0%})")


if __name__ == "__main__":
    main()
