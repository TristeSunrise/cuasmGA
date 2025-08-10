# safe_mem_mover.py
# 只移动“内存类指令”的相邻安全交换工具（无 GA 依赖）

from typing import List, Optional, Callable, Dict, Set, Tuple
import random

from static_analysis import static_analysis
from gpu_utils import (
    get_gpu_cc, get_mutatable_ops, get_st_window, get_min_stall_count,
    check_adj_opcodes, get_st_database
)

# ---- 轻量适配器：把两个函数包装成旧接口 ----
import decoder


class SafeMemMover:
    """
    仅做“内存指令”的相邻安全交换：
      - 候选发现：static_analysis（MEMORY_OPS / BAN_OPS）
      - 掩码判定：RAW/WAR、check_adj_opcodes、scoreboard waits、最小 stall 窗口
      - 动作：只与相邻一行交换
    """

    def __init__(
        self,
        decoder,
        *,
        cc: Optional[Tuple[int, int]] = None,
        memory_ops: Optional[Tuple[str, ...]] = None,
        ban_ops: Optional[Tuple[str, ...]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.decoder = decoder
        self.cc = cc if cc is not None else get_gpu_cc()
        if memory_ops is None or ban_ops is None:
            mops, bops = get_mutatable_ops(self.cc)
            if memory_ops is None: memory_ops = tuple(mops)
            if ban_ops is None:    ban_ops    = tuple(bops)
        self.memory_ops = tuple(memory_ops)
        self.ban_ops    = tuple(ban_ops)
        self.st_window  = get_st_window(self.cc)
        self.st_db      = get_st_database(self.cc)
        self.rng        = rng or random.Random()

        # 每次 candidates() 会写入/更新
        self.min_st_analysis: Dict[str, int] = {}
        self.black_list: Set[str] = set()

    # ---------- 对外接口 ----------
    def candidates(self, sass: List[str]) -> List[int]:
        """返回可移动的“内存指令”行号列表（并更新 min_st_analysis / black_list）"""
        self.min_st_analysis.clear()
        self.black_list.clear()
        cands, *_ = static_analysis(
            sass, self.decoder, self.ban_ops, self.memory_ops,
            self.min_st_analysis, self.black_list, self.st_db
        )
        return cands

    def mask(self, sass: List[str], lineno: int) -> List[int]:
        """返回 [can_move_up, can_move_down]"""
        return self._gen_mask_for_line(sass, lineno)

    def step(self, sass: List[str], *, max_trials: int = 20) -> bool:
        """
        在 sass 上“尝试一次随机安全相邻交换”，成功则原地修改并返回 True，否则 False。
        """
        cands = self.candidates(sass)
        if not cands:
            return False

        order = list(range(len(cands)))
        self.rng.shuffle(order)
        tried = 0
        while order and tried < max_trials:
            idx = order.pop()
            lineno = cands[idx]
            up, down = self.mask(sass, lineno)
            allowed = []
            if up:   allowed.append(0)
            if down: allowed.append(1)
            if not allowed:
                tried += 1
                continue
            self._swap_adjacent_in_place(sass, lineno, self.rng.choice(allowed))
            return True
        return False

    # ---------- 内部实现 ----------
    def _gen_mask_for_line(self, kernel_section: List[str], lineno: int) -> List[int]:
        line = kernel_section[lineno].strip()
        ctrl_code, _, predicate, opcode, dst, src, meta = self.decoder.decode(line)
        p_st = None
        
        if ctrl_code is None:
            return [0, 0]

        N = len(kernel_section)
        up_ok   = (lineno > 0)
        down_ok = (lineno < N - 1)

        waits, r, w, _, self_stall_str = self.decoder.decode_ctrl_code(ctrl_code)
        r = -1 if r[1] == '-' else int(r[1])
        w = -1 if w[1] == '-' else int(w[1])

        # ---- 上移检查 ----
        if up_ok:
            prev = kernel_section[lineno - 1].strip()
            p_ctrl, _, p_pred, p_op, p_dst, p_src, p_meta = self.decoder.decode(prev)
            if p_ctrl is None:
                up_ok = False
            elif (p_dst in (src or [])) or (dst in (p_src or [])):
                up_ok = False
            elif not check_adj_opcodes(self.cc, p_op or "", opcode or "",
                                       p_dst, dst, p_src or [], src or [],
                                       p_pred, predicate, p_meta):
                up_ok = False
            else:
                for bo in self.ban_ops:
                    if (p_op or "").startswith(bo):
                        up_ok = False
                        break
                if up_ok:
                    p_waits, pr, pw, _, p_st = self.decoder.decode_ctrl_code(p_ctrl)
                    pr = -1 if pr[1] == '-' else int(pr[1])
                    pw = -1 if pw[1] == '-' else int(pw[1])
                    if (pr in waits) or (pw in waits) or (r in p_waits) or (w in p_waits):
                        up_ok = False

                # stall 窗口
                if up_ok and p_st is not None:
                    total = int(p_st[1:-1])
                    for i in range(1, 1 + self.st_window):
                        j = lineno + i
                        if j >= N: break
                        t_ctrl, *_, t_op, t_dst, t_src, _ = self.decoder.decode(kernel_section[j].strip())
                        if t_ctrl is None:
                            continue
                        _, _, _, _, t_st = self.decoder.decode_ctrl_code(t_ctrl)
                        min_st = get_min_stall_count(self.cc, p_op or "", t_op or "")
                        if (p_op or "") in self.min_st_analysis:
                            min_st = self.min_st_analysis[p_op]
                        if p_dst is not None and (p_dst in (t_src or [])) and total <= min_st:
                            up_ok = False
                            break
                        total += int(t_st[1:-1])

                    if up_ok:
                        total = 0
                        for i in range(2, 2 + self.st_window):
                            j = lineno - i
                            if j < 0: break
                            t_ctrl, *_, t_op, t_dst, t_src, _ = self.decoder.decode(kernel_section[j].strip())
                            if t_ctrl is None:
                                continue
                            _, _, _, _, t_st = self.decoder.decode_ctrl_code(t_ctrl)
                            total += int(t_st[1:-1])
                            min_st = get_min_stall_count(self.cc, opcode or "", t_op or "")
                            if t_op in self.min_st_analysis:
                                min_st = self.min_st_analysis[t_op]
                            if t_dst is not None and (t_dst in (src or [])) and total <= min_st:
                                up_ok = False
                                break

        # ---- 下移检查 ----
        if down_ok:
            nxt = kernel_section[lineno + 1].strip()
            n_ctrl, _, n_pred, n_op, n_dst, n_src, n_meta = self.decoder.decode(nxt)
            if n_ctrl is None:
                down_ok = False
            elif (dst in (n_src or [])) or (n_dst in (src or [])):
                down_ok = False
            elif not check_adj_opcodes(self.cc, opcode or "", n_op or "",
                                       dst, n_dst, src or [], n_src or [],
                                       predicate, n_pred, n_meta):
                down_ok = False
            else:
                for bo in self.ban_ops:
                    if (n_op or "").startswith(bo):
                        down_ok = False
                        break
                if down_ok:
                    n_waits, *_ = self.decoder.decode_ctrl_code(n_ctrl)
                    if (r in n_waits) or (w in n_waits):
                        down_ok = False

        return [1 if up_ok else 0, 1 if down_ok else 0]

    @staticmethod
    def _swap_adjacent_in_place(kernel_section: List[str], lineno: int, direction: int) -> int:
        """
        direction=0 上移（与上一行交换），1 下移（与下一行交换）
        返回交换后该指令的新行号
        """
        if direction == 0 and lineno > 0:
            kernel_section[lineno - 1], kernel_section[lineno] = kernel_section[lineno], kernel_section[lineno - 1]
            return lineno - 1
        if direction == 1 and lineno < len(kernel_section) - 1:
            kernel_section[lineno], kernel_section[lineno + 1] = kernel_section[lineno + 1], kernel_section[lineno]
            return lineno + 1
        return lineno
