# sass_dag_builder.py  —— 纯函数版（无文件IO）
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re

# ---- 基础正则 ----
UR_RE  = re.compile(r"\bUR(\d+)\b")
R_RE   = re.compile(r"\bR(\d+)\b")
P_RE   = re.compile(r"\bP(\d+)\b")
CC_RE  = re.compile(r"\b(?:CC|X)\b")
BRACKET_RE = re.compile(r"\[(.*?)\]")

# ---- 可选清洗：去掉 /*...*/ 与 // 注释，收紧空白行 ----
def clean_sass_lines(lines: List[str], *, keep_comments: bool = False) -> List[str]:
    out: List[str] = []
    for line in lines:
        s = line
        if not keep_comments:
            s = re.sub(r"/\*.*?\*/", "", s)  # 去地址标注等注释
            s = re.sub(r"//.*", "", s)
        s = s.strip()
        if s:
            out.append(s)
    return out

# ---- 指令分类/内存空间判断（保守）----
def classify_mnemonic(op: str) -> str:
    u = op.upper()
    if u.startswith("LD"): return "LOAD"
    if u.startswith("ST"): return "STORE"
    if u.startswith("ATOM") or u.startswith("RED"): return "ATOMIC"
    if "MEMBAR" in u or "BAR.SYNC" in u or "DEPBAR" in u or "LDGDEPBAR" in u: return "BARRIER"
    if "COMMIT_GROUP" in u or "WAIT_GROUP" in u or ("CP" in u and "ASYNC" in u) or "LDGSTS" in u: return "CPASYNC"
    if u in ("BRA", "BRX", "JMP") or u.startswith("SSY") or u.startswith("PBK") or u.startswith("BRK"): return "CTRL_FLOW"
    if u in ("RET", "EXIT"): return "CTRL_FLOW"
    return "ALU"

def detect_space(op: str) -> str:
    u = op.upper()
    if "LDS" in u or ".SHARED" in u: return "SHARED"
    if "LDG" in u or ".GLOBAL" in u or "ATOM" in u or "RED" in u: return "GLOBAL"
    if ".LOCAL" in u: return "LOCAL"
    if "LDGSTS" in u: return "GLOBAL|SHARED"
    return "UNKNOWN"

def parse_regs_in_brackets(operand: str) -> Set[str]:
    regs: Set[str] = set()
    for m in BRACKET_RE.finditer(operand):
        inside = m.group(1)
        for r in R_RE.findall(inside): regs.add(f"R{r}")
        for ur in UR_RE.findall(inside): regs.add(f"UR{ur}")
        for p in P_RE.findall(inside): regs.add(f"P{p}")
    return regs

def _unwrap_regs(x: str) -> List[str]:
    if "{" in x and "}" in x:
        inside = x[x.find("{")+1:x.rfind("}")]
        return [r.strip() for r in inside.split(",") if r.strip()]
    return [x]

# ---- 逐行解析：读/写寄存器、内存读写键、屏障/控制流 ----
def parse_line_rw(line: str) -> dict:
    original = line
    pred_uses: Set[str] = set()

    s = line.strip()
    if s.startswith("@"):  # @P0 / @!P0
        m = re.match(r"@!?P(\d+)\s+(.*)", s, flags=re.IGNORECASE)
        if m:
            pred_uses.add(f"P{m.group(1)}")
            s = m.group(2).strip()

    parts = re.split(r"\s+", s, maxsplit=1)
    op = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    tmp = re.sub(r"\{([^}]*)\}", lambda m: "{" + m.group(1).replace(",", ";") + "}", rest)
    ops = [o.strip().replace(";", ",") for o in tmp.split(",") if o.strip()]

    klass = classify_mnemonic(op)
    space = detect_space(op)

    reads: Set[str] = set()
    writes: Set[str] = set()
    mem_reads: List[Tuple[str, str]] = []
    mem_writes: List[Tuple[str, str]] = []

    # 谓词前缀算读
    reads |= pred_uses

    bracket_regs = parse_regs_in_brackets(rest)

    def collect_regs(token: str, as_writes: bool = False):
        tgt = writes if as_writes else reads
        for r in R_RE.findall(token):
            if f"R{r}" != "RZ": tgt.add(f"R{r}")
        for ur in UR_RE.findall(token):
            if f"UR{ur}" != "URZ": tgt.add(f"UR{ur}")
        for p in P_RE.findall(token): tgt.add(f"P{p}")
        if "CC" in token or "X" in token:
            tgt.add("CC"); tgt.add("X")

    if re.match(r"(PSETP|ISETP|FSETP)", op, re.IGNORECASE):
        # 谓词目的写
        if ops:
            for tok in _unwrap_regs(ops[0]):
                if re.match(r"P\d+\b", tok): writes.add(tok)
        if len(ops) > 1:
            for tok in _unwrap_regs(ops[1]):
                if re.match(r"P\d+\b", tok): writes.add(tok)
        for t in ops[2:]:
            for tok in _unwrap_regs(t): collect_regs(tok, as_writes=False)

    elif klass == "LOAD":
        before = rest.split("[", 1)[0]
        for tok in _unwrap_regs(before): collect_regs(tok, as_writes=True)  # 目的寄存器
        for r in bracket_regs:
            if r not in ("RZ","URZ"): reads.add(r)  # 地址寄存器读
        base = next((x for x in bracket_regs if x.startswith(("R","UR"))), "ANY")
        mem_reads.append((space, base))

    elif klass in ("STORE", "ATOMIC"):
        for t in ops:
            for tok in _unwrap_regs(t): collect_regs(tok, as_writes=False)
        base = next((x for x in bracket_regs if x.startswith(("R","UR"))), "ANY")
        if klass == "ATOMIC": mem_reads.append((space, base))
        mem_writes.append((space, base))

    else:  # ALU等
        if ops:
            for tok in _unwrap_regs(ops[0]): collect_regs(tok, as_writes=True)
            for t in ops[1:]:
                for tok in _unwrap_regs(t): collect_regs(tok, as_writes=False)
        for r in bracket_regs: reads.add(r)

    is_barrier = (classify_mnemonic(op) in ("BARRIER", "CPASYNC"))
    is_ctrl    = (classify_mnemonic(op) == "CTRL_FLOW")

    return {
        "op": op,
        "reads": sorted(reads),
        "writes": sorted(writes),
        "mem_reads": mem_reads,
        "mem_writes": mem_writes,
        "space": space,
        "is_barrier": is_barrier,
        "is_ctrl": is_ctrl,
        "text": original.strip(),
    }

# ---- 构图：保守 RAW/WAR/WAW + 粗别名内存；屏障/控制流切段强顺序 ----
def build_preds(parsed: List[dict], *, split_on_barrier: bool = True) -> Dict[int, Set[int]]:
    N = len(parsed)
    preds: Dict[int, Set[int]] = {i: set() for i in range(N)}

    # 1) 屏障/控制流分段：跨段强顺序（禁止重排）
    phase_of = [0] * N
    if split_on_barrier:
        phase = 0
        for i, info in enumerate(parsed):
            phase_of[i] = phase
            if info["is_barrier"] or info["is_ctrl"]:
                phase += 1
        by_phase = defaultdict(list)
        for i, ph in enumerate(phase_of):
            by_phase[ph].append(i)
        phases = sorted(by_phase.keys())
        for idx in range(1, len(phases)):
            prev_items = [j for p in phases[:idx] for j in by_phase[p]]
            cur_items  = by_phase[phases[idx]]
            for u in prev_items:
                for v in cur_items:
                    preds[v].add(u)

    # 2) 段内寄存器依赖与内存依赖
    last_writer: Dict[str, int] = {}
    last_readers: Dict[str, Set[int]] = {}
    last_mem_write: Dict[Tuple[str,str], int] = {}
    last_mem_readers: Dict[Tuple[str,str], Set[int]] = {}

    for i, info in enumerate(parsed):
        # 寄存器：RAW/WAR/WAW
        uses = set(info["reads"])
        defs = set(info["writes"])

        for r in uses:
            if r in last_writer:
                preds[i].add(last_writer[r])      # RAW
        for r in defs:
            if r in last_writer:
                preds[i].add(last_writer[r])      # WAW
            if r in last_readers:
                for rd in last_readers[r]:
                    preds[i].add(rd)              # WAR
            last_writer[r] = i
            last_readers[r] = set()
        for r in uses:
            last_readers.setdefault(r, set()).add(i)

        # 内存：按 (space, base_reg or "ANY") 粗别名；ATOMIC 视为读+写
        def mem_keys(mem_list, space_default):
            keys = []
            for (space_tag, base) in mem_list:
                st = space_tag if space_tag != "UNKNOWN" else space_default
                keys.append((st, base))
            return keys

        read_keys  = mem_keys(info["mem_reads"],  info["space"])
        write_keys = mem_keys(info["mem_writes"], info["space"])

        for k in read_keys:
            if k in last_mem_write:
                preds[i].add(last_mem_write[k])   # 读依赖最近写
            last_mem_readers.setdefault(k, set()).add(i)

        for k in write_keys:
            if k in last_mem_write:
                preds[i].add(last_mem_write[k])   # WAW
            for rd in last_mem_readers.get(k, set()):
                preds[i].add(rd)                  # WAR
            last_mem_write[k] = i
            last_mem_readers[k] = set()

    return preds

# ---- 入口：直接吃 list[str]，返回 (baseline_sass, preds) ----
def build_from_lines(sass_lines: List[str], *,
                     keep_comments: bool = False,
                     split_on_barrier: bool = True) -> Tuple[List[str], Dict[int, Set[int]]]:
    """
    输入：sass_lines = 内核SASS的行列表（list[str]）
    输出：baseline_sass（清洗后的文本行）与 preds（依赖图，index->前驱集合）
    """
    baseline = clean_sass_lines(sass_lines, keep_comments=keep_comments)
    parsed = [parse_line_rw(line) for line in baseline]
    preds = build_preds(parsed, split_on_barrier=split_on_barrier)
    return baseline, preds
