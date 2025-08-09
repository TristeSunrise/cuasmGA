from sass_dag_builder import clean_sass_lines, parse_line_rw

def build_movable_mask(pure_kernel_lines: list[str]) -> list[bool]:
    # 用与构图相同的清洗（去方括号+去地址）保证索引一致
    baseline_clean = clean_sass_lines(
        pure_kernel_lines, strip_flags=True, strip_addr_comments=True
    )
    parsed = [parse_line_rw(s) for s in baseline_clean]

    def is_movable(info: dict) -> bool:
        # 仅 ALU
        if info["klass"] != "ALU":
            return False
        # 不触碰谓词 / 条件码
        if any(x.startswith("P") for x in info["reads"] + info["writes"]):
            return False
        if ("CC" in info["reads"] or "CC" in info["writes"] or
            "X"  in info["reads"] or "X"  in info["writes"]):
            return False
        # 不涉及任何内存键
        if info["mem_reads"] or info["mem_writes"]:
            return False
        # 排除读特殊寄存器的指令（更稳）
        u = info["op"].upper()
        if u in ("S2R", "CS2R"):
            return False
        return True

    mask = [is_movable(p) for p in parsed]
    # 兜底：如果全是 False，就把“非常安全”的算术列为 True（极少见）
    if not any(mask):
        SAFE = ("MOV","IADD","IADD3","IMAD","FADD","FMUL","FFMA","LOP3","SHF","SHL","SHR","PRMT")
        for i, p in enumerate(parsed):
            u = p["op"].upper().split('.')[0]
            if p["klass"] == "ALU" and u in SAFE and not (p["mem_reads"] or p["mem_writes"]):
                mask[i] = True
    return mask
