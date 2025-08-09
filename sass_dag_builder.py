# sass_dag_builder.py — STRICT (no file I/O)
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re

# ----------------- Regex -----------------
UR_RE   = re.compile(r"\bUR(\d+)\b")
R_RE    = re.compile(r"\bR(\d+)\b")
P_RE    = re.compile(r"\bP(\d+)\b")
BRK_RE  = re.compile(r"\[(.*?)\]")                 # address brackets [...]
FLAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*")         # leading scheduling flags [B...]
ADDR_RE = re.compile(r"/\*[^*]*\*/")               # /*00f0*/ style addresses

# ----------------- Cleaning -----------------
def clean_sass_lines(
    lines: List[str],
    *,
    strip_flags: bool = True,
    strip_addr_comments: bool = True,
) -> List[str]:
    """
    仅用于构图（DAG），不是写回。建议：strip_flags=True，strip_addr_comments=True。
    写回 cubin 时请使用“保留地址、无方括号”的版本。
    """
    out: List[str] = []
    for line in lines:
        s = line
        if strip_flags:
            s = FLAG_RE.sub("", s)
        if strip_addr_comments:
            s = ADDR_RE.sub("", s)
        s = re.sub(r"//.*", "", s)   # end-of-line comments
        s = s.strip()
        if not s:
            continue
        # 保证以分号结尾（有些反汇编行可能漏 ';'）
        if not s.endswith(";"):
            s += " ;"
        out.append(s)
    return out

# ----------------- Classifiers -----------------
def classify_mnemonic(op: str) -> str:
    u = op.upper()
    if u.startswith("LD"): return "LOAD"
    if u.startswith("ST"): return "STORE"
    if u.startswith("ATOM") or u.startswith("RED"): return "ATOMIC"
    if any(x in u for x in ("MEMBAR", "BAR.SYNC", "DEPBAR", "LDGDEPBAR")): return "BARRIER"
    if ("COMMIT_GROUP" in u) or ("WAIT_GROUP" in u) or ("LDGSTS" in u) or ("CP" in u and "ASYNC" in u): return "CPASYNC"
    if u in ("RET", "EXIT") or u in ("BRA", "BRX", "JMP") or u.startswith(("SSY","PBK","BRK")): return "CTRL_FLOW"
    # Tensor/Matrix hints（严格处理时将其视为“非ALU锚点”）
    if u.startswith(("HMMA","IMMA","WGMMA","LDMATRIX","LDSM")): return "TENSOR"
    return "ALU"

def detect_space(op: str) -> str:
    u = op.upper()
    if "LDS" in u or ".SHARED" in u or u.startswith(("LDMATRIX","LDSM")): return "SHARED"
    if "LDG" in u or ".GLOBAL" in u or "ATOM" in u or "RED" in u: return "GLOBAL"
    if ".LOCAL" in u: return "LOCAL"
    if "LDGSTS" in u: return "GLOBAL|SHARED"  # global->shared path
    return "UNKNOWN"

def _unwrap_regs(x: str) -> List[str]:
    # keep {...} groups as multiple regs
    if "{" in x and "}" in x:
        inside = x[x.find("{")+1:x.rfind("}")]
        return [r.strip() for r in inside.split(",") if r.strip()]
    return [x]

def parse_regs_in_brackets(operand: str) -> Set[str]:
    regs: Set[str] = set()
    for m in BRK_RE.finditer(operand):
        inside = m.group(1)
        for r in R_RE.findall(inside):  regs.add(f"R{r}")
        for ur in UR_RE.findall(inside): regs.add(f"UR{ur}")
        for p in P_RE.findall(inside): regs.add(f"P{p}")
    return regs

# ----------------- Parser (reads/writes/mem) -----------------
def parse_line_rw(line: str) -> dict:
    original = line
    pred_uses: Set[str] = set()

    s = line.strip()
    # predicate prefix: @P0 / @!P0
    if s.startswith("@"):
        m = re.match(r"@!?P(\d+)\s+(.*)", s, flags=re.IGNORECASE)
        if m:
            pred_uses.add(f"P{m.group(1)}")
            s = m.group(2).strip()

    parts = re.split(r"\s+", s, maxsplit=1)
    op   = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    tmp = re.sub(r"\{([^}]*)\}", lambda m: "{" + m.group(1).replace(",", ";") + "}", rest)
    ops = [o.strip().replace(";", ",") for o in tmp.split(",") if o.strip()]

    klass = classify_mnemonic(op)
    space = detect_space(op)

    reads: Set[str] = set(pred_uses)
    writes: Set[str] = set()
    mem_reads: List[Tuple[str,str]]  = []
    mem_writes: List[Tuple[str,str]] = []

    bracket_regs = parse_regs_in_brackets(rest)

    def collect_regs(token: str, as_writes: bool = False):
        tgt = writes if as_writes else reads
        for r in R_RE.findall(token):
            if f"R{r}" != "RZ": tgt.add(f"R{r}")
        for ur in UR_RE.findall(token):
            if f"UR{ur}" != "URZ": tgt.add(f"UR{ur}")
        for p in P_RE.findall(token):
            tgt.add(f"P{p}")
        # 条件码/进位（严格：一律按读处理；在第一目的位出现时按写）
        if "CC" in token or " X" in f" {token}":
            tgt.add("CC"); tgt.add("X")

    # Predicate-set write
    if re.match(r"(PSETP|ISETP|FSETP)", op, re.IGNORECASE):
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
        for tok in _unwrap_regs(before):
            collect_regs(tok, as_writes=True)  # dest regs
        for r in bracket_regs:
            if r not in ("RZ","URZ"): reads.add(r)
        base = next((x for x in bracket_regs if x.startswith(("R","UR"))), "ANY")
        mem_reads.append((space, base))

    elif klass in ("STORE", "ATOMIC"):
        for t in ops:
            for tok in _unwrap_regs(t): collect_regs(tok, as_writes=False)
        base = next((x for x in bracket_regs if x.startswith(("R","UR"))), "ANY")
        if klass == "ATOMIC": mem_reads.append((space, base))
        mem_writes.append((space, base))

    else:  # ALU/TENSOR/CTRL/BARRIER/CPASYNC (寄存器)
        if ops:
            for tok in _unwrap_regs(ops[0]): collect_regs(tok, as_writes=True)
            for t in ops[1:]:
                for tok in _unwrap_regs(t): collect_regs(tok, as_writes=False)
        for r in bracket_regs: reads.add(r)

    is_barrier = (klass in ("BARRIER",))
    is_ctrl    = (klass in ("CTRL_FLOW",))

    return {
        "op": op,
        "klass": klass,
        "reads": sorted(reads),
        "writes": sorted(writes),
        "mem_reads": mem_reads,
        "mem_writes": mem_writes,
        "space": space,
        "is_barrier": is_barrier,
        "is_ctrl": is_ctrl,
        "text": original.strip(),
    }

# ----------------- Strict DAG builder -----------------
def build_preds(parsed: List[dict], *, split_on_barrier: bool = True) -> Dict[int, Set[int]]:
    """
    严格版：
      - 屏障/控制流切段：跨段强顺序
      - RAW/WAR/WAW + 粗别名内存
      - cp.async/LDGSTS → COMMIT/LDGDEPBAR → DEPBAR/WAIT → SHARED消费者 的状态机依赖
      - 非ALU锚点串链（保持相对顺序）
      - 段内所有 SHARED 访存串链，并依赖最近 WAIT/BAR/MEMBAR
    """
    N = len(parsed)
    preds: Dict[int, Set[int]] = {i: set() for i in range(N)}

    # ---- 1) 分段：跨段强顺序 ----
    phase_of = [0]*N
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

    # ---- 2) 段内寄存器/内存基本依赖 ----
    last_writer: Dict[str, int] = {}
    last_readers: Dict[str, Set[int]] = {}
    last_mem_write: Dict[Tuple[str,str], int] = {}
    last_mem_readers: Dict[Tuple[str,str], Set[int]] = {}

    # 额外辅助：本段非ALU锚点、共享访存列表
    phase_anchors: Dict[int, List[int]] = defaultdict(list)
    phase_shared:  Dict[int, List[int]] = defaultdict(list)

    # cp.async 状态机跟踪（逐全体顺序，但遇屏障/控制流会复位）
    open_cp: List[int] = []   # 生产者：LDGSTS/CP.ASYNC*
    last_commit: int | None = None
    last_wait:   int | None = None

    def is_cp_async(uop: str) -> bool:
        return ("LDGSTS" in uop) or ("CP" in uop and "ASYNC" in uop)

    def is_commit(uop: str) -> bool:
        return ("LDGDEPBAR" in uop) or ("COMMIT_GROUP" in uop)

    def is_wait(uop: str) -> bool:
        return ("DEPBAR" in uop) or ("WAIT_GROUP" in uop)

    def is_tensor_or_nonalu(info: dict) -> bool:
        k = info["klass"]
        return k in ("LOAD","STORE","ATOMIC","CPASYNC","BARRIER","CTRL_FLOW","TENSOR")

    for i, info in enumerate(parsed):
        ph = phase_of[i]
        uop = info["op"].upper()

        # ---- 基本 RAW/WAR/WAW ----
        uses = set(info["reads"])
        defs = set(info["writes"])
        for r in uses:
            if r in last_writer:
                preds[i].add(last_writer[r])  # RAW
        for r in defs:
            if r in last_writer:
                preds[i].add(last_writer[r])  # WAW
            if r in last_readers:
                for rd in last_readers[r]:
                    preds[i].add(rd)          # WAR
            last_writer[r] = i
            last_readers[r] = set()
        for r in uses:
            last_readers.setdefault(r, set()).add(i)

        # ---- 粗别名内存 ----
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
                preds[i].add(last_mem_write[k])   # 写依赖最近写（WAW）
            for rd in last_mem_readers.get(k, set()):
                preds[i].add(rd)                  # WAR
            last_mem_write[k] = i
            last_mem_readers[k] = set()

        # ---- 记录锚点 & 共享访存 ----
        if is_tensor_or_nonalu(info):
            phase_anchors[ph].append(i)
        if (info["space"] == "SHARED") or info["op"].upper().startswith(("LDMATRIX","LDSM")):
            phase_shared[ph].append(i)

        # ---- cp.async 状态机约束 ----
        if is_cp_async(uop):
            open_cp.append(i)

        if is_commit(uop):
            for u in open_cp:
                preds[i].add(u)          # 生产者 -> commit
            last_commit = i

        if is_wait(uop):
            if last_commit is not None:
                preds[i].add(last_commit)  # commit -> wait
            else:
                for u in open_cp:
                    preds[i].add(u)        # 无 commit 时：生产者 -> wait
            last_wait = i
            # 结束当前组
            open_cp.clear()
            last_commit = None

        # wait 之后才允许共享内存消费者继续（保守）
        if ((info["space"] == "SHARED") or uop.startswith(("LDMATRIX","LDSM"))) and (last_wait is not None):
            preds[i].add(last_wait)

        # 硬边界重置
        if info["is_ctrl"] or info["is_barrier"]:
            open_cp.clear()
            last_commit = None
            last_wait = None

    # ---- 3) 锚点串链（非ALU指令保持相对顺序） ----
    for ph, anchors in phase_anchors.items():
        for a, b in zip(anchors, anchors[1:]):
            preds[b].add(a)

    # ---- 4) 共享访存串链（更保守：同段内 SHARED 全按顺序相连） ----
    for ph, sh in phase_shared.items():
        for a, b in zip(sh, sh[1:]):
            preds[b].add(a)

    return preds

# ----------------- Public API -----------------
def build_from_lines(
    sass_lines: List[str],
    *,
    strip_flags: bool = True,
    strip_addr_comments: bool = True,
    split_on_barrier: bool = True,
) -> Tuple[List[str], Dict[int, Set[int]]]:
    """
    输入：原始 SASS 行（list[str]）。本函数仅用于“构图/重排”，不是写回。
    - strip_flags=True：去掉行首的 [B...:R-:W-:...:Sxx] 调度标记（建议开）
    - strip_addr_comments=True：去掉 /*00f0*/ 地址注释（建议开；写回请使用保留地址的版本）
    - split_on_barrier=True：遇到 CTRL/BARRIER 时分段，跨段强顺序
    返回：baseline（清洗后的行，用作 GA 的 baseline_text）与 preds（DAG，index->前驱集合）。
    """
    baseline = clean_sass_lines(
        sass_lines,
        strip_flags=strip_flags,
        strip_addr_comments=strip_addr_comments,
    )
    parsed = [parse_line_rw(line) for line in baseline]
    preds  = build_preds(parsed, split_on_barrier=split_on_barrier)
    return baseline, preds
