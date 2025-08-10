# safe_mem_mover.py
# 只移动“内存类指令”的相邻安全交换工具（尽量贴近原 Sample 逻辑）

from typing import List, Optional, Dict, Set, Tuple
import random

from static_analysis import static_analysis
from gpu_utils import (
    get_gpu_cc, get_mutatable_ops, get_st_window, get_min_stall_count,
    check_adj_opcodes, get_st_database
)
from decoder import decode, decode_ctrl_code  # 需要提供: decode(line), decode_ctrl_code(ctrl)


class SafeMemMover:
    """
    与原 Sample 保持一致的判定链：
      - 候选发现：static_analysis（使用 MEMORY_OPS / BAN_OPS）
      - 掩码：RAW/WAR、check_adj_opcodes、scoreboard waits、最小 stall 窗口
      - 行为：仅相邻交换（上/下）
      - 复用：跨调用复用 min_st_analysis / black_list，并在每次分析后做 ST_DB 合并 + 剪枝
    """

    def __init__(
        self,
        *,
        cc: Optional[Tuple[int, int]] = None,
        memory_ops: Optional[Tuple[str, ...]] = None,
        ban_ops: Optional[Tuple[str, ...]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.cc = cc if cc is not None else get_gpu_cc()

        # 与 get_mutatable_ops 保持一致（可传入覆盖）
        if memory_ops is None or ban_ops is None:
            mops, bops = get_mutatable_ops(self.cc)
            if memory_ops is None:
                memory_ops = tuple(mops)
            if ban_ops is None:
                ban_ops = tuple(bops)
        self.memory_ops = tuple(memory_ops)
        self.ban_ops = tuple(ban_ops)

        self.st_window = get_st_window(self.cc)
        self.st_db = get_st_database(self.cc)
        self.rng = rng or random.Random()

        # 持久化复用（不要每次都清空）
        self.min_st_analysis: Dict[str, int] = {}
        self.black_list: Set[str] = set()

        # 可选：用于检测是否换了完全不同的 kernel；如需强制 reset 可用 reset()
        self._kernel_sig: Optional[int] = None

    # ---------- 维护 ----------
    def reset(self):
        """当你切换到一个完全不同的 kernel 时，可手动 reset。"""
        self.min_st_analysis.clear()
        self.black_list.clear()
        self._kernel_sig = None

    def _merge_and_prune_min_st(self):
        # 与原 Sample.static_analysis() 的后处理一致
        # 1) 用 ST_DB 收紧（取更小的经验下界）
        for k, v in list(self.min_st_analysis.items()):
            if k in self.st_db:
                self.min_st_analysis[k] = min(self.st_db[k], v)
        # 2) 剪掉过大的 & LDS*
        for k, v in list(self.min_st_analysis.items()):
            if v > 20 or k.startswith('LDS'):
                del self.min_st_analysis[k]

    # ---------- 对外接口 ----------
    def candidates(self, sass: List[str], *, refresh: bool = True) -> List[int]:
        """
        返回可移动的“内存指令”行号列表。
        - refresh=True：根据当前 sass 做一次增量分析/更新（默认）
        - 复用 self.min_st_analysis / self.black_list，不每次清空
        """
        if refresh:
            sig = hash('\n'.join(sass))
            if self._kernel_sig is None:
                self._kernel_sig = sig
            elif sig != self._kernel_sig:
                # 核心序列大改（换 kernel），重置；如果只是小范围重排也会变，但仍可复用；
                # 若你不想重置，可把下面两行注释掉。
                # self.reset()
                self._kernel_sig = sig

            cands, *_ = static_analysis(
                sass, self.ban_ops, self.memory_ops,
                self.min_st_analysis, self.black_list, self.st_db
            )
            # 合并 + 剪枝，保持与原逻辑一致
            self._merge_and_prune_min_st()
            return cands

        # 不刷新，直接返回上次分析得到的 candidates（这里简单起见重新分析一次）
        cands, *_ = static_analysis(
            sass,  self.ban_ops, self.memory_ops,
            self.min_st_analysis, self.black_list, self.st_db
        )
        self._merge_and_prune_min_st()
        return cands

    def mask(self, sass: List[str], lineno: int) -> List[int]:
        """返回 [can_move_up, can_move_down]（严格复刻 Sample 的 _generate_mask）"""
        return self._gen_mask_for_line(sass, lineno)

    def step(self, sass: List[str], *, max_trials: int = 1) -> bool:
        cands = self.candidates(sass, refresh=True)
        if not cands:
            return False

        order = list(range(len(cands)))
        self.rng.shuffle(order)
        tried = 0
        while order and tried < max_trials:
            idx = order.pop()
            lineno = cands[idx]
            up, _down = self.mask(sass, lineno)   # 只看上移
            if not up:
                tried += 1
                continue
            # 上移：与上一行交换
            self._swap_adjacent_in_place(sass, lineno, 0)
            return True
        return False

    # ---------- 内部：与 Sample._generate_mask 保持一致 ----------
    def _gen_mask_for_line(self, kernel_section: List[str], lineno: int) -> List[int]:
        line = kernel_section[lineno].strip()
        ctrl, _, pred, op, dst, src, meta = decode(line)
        if ctrl is None:
            # label 等，不可跨
            return [0, 0]

        N = len(kernel_section)
        up_ok = (lineno > 0)
        down_ok = (lineno < N - 1)

        waits, r, w, _, self_st = decode_ctrl_code(ctrl)
        r = -1 if r[1] == '-' else int(r[1])
        w = -1 if w[1] == '-' else int(w[1])

        # ---------- 上移检查 ----------
        if up_ok:
            prev = kernel_section[lineno - 1].strip()
            p_ctrl, _, p_pred, p_op, p_dst, p_src, p_meta = decode(prev)
            if p_ctrl is None:
                up_ok = False
            # 直接依赖：RAW / WAR
            elif (p_dst in (src or [])) or (dst in (p_src or [])):
                up_ok = False
            # 语义邻接规则
            elif not check_adj_opcodes(
                self.cc,
                p_op or "",
                op or "",
                p_dst,
                dst,
                p_src or [],
                src or [],
                p_pred,
                pred,
                p_meta,
            ):
                up_ok = False
            else:
                # 邻居禁令
                for bo in self.ban_ops:
                    if (p_op or "").startswith(bo):
                        up_ok = False
                        break

                # scoreboard（barrier 槽）
                if up_ok:
                    p_waits, pr, pw, _, p_st = decode_ctrl_code(p_ctrl)
                    pr = -1 if pr[1] == '-' else int(pr[1])
                    pw = -1 if pw[1] == '-' else int(pw[1])
                    if (pr in waits) or (pw in waits) or (r in p_waits) or (w in p_waits):
                        up_ok = False

                # stall 窗口（两段）
                if up_ok and p_st is not None:
                    # (1) prev -> 向后看窗口，若某条使用了 p_dst，且累计 stall 不足最小值，则不能把当前内存指令拉到它前面
                    total = int(p_st[1:-1])
                    for i in range(1, 1 + self.st_window):
                        j = lineno + i
                        if j >= N:
                            break
                        t_ctrl, *_, t_op, t_dst, t_src, _ = decode(kernel_section[j].strip())
                        if t_ctrl is None:
                            continue
                        *_, t_st = decode_ctrl_code(t_ctrl)
                        min_st = get_min_stall_count(self.cc, p_op or "", t_op or "")
                        if (p_op or "") in self.min_st_analysis:
                            min_st = self.min_st_analysis[p_op]
                        if p_dst is not None and (p_dst in (t_src or [])) and total <= min_st:
                            up_ok = False
                            break
                        total += int(t_st[1:-1])

                    # (2) cur(mem) -> 向前看窗口，若窗口内有写出寄存器为 cur 源寄存器的指令，且累计 stall 不足最小值，也不许上移
                    if up_ok:
                        total = 0
                        for i in range(2, 2 + self.st_window):
                            j = lineno - i
                            if j < 0:
                                break
                            t_ctrl, *_, t_op, t_dst, t_src, _ = decode(kernel_section[j].strip())
                            if t_ctrl is None:
                                continue
                            *_, t_st = decode_ctrl_code(t_ctrl)
                            total += int(t_st[1:-1])
                            min_st = get_min_stall_count(self.cc, op or "", t_op or "")
                            if t_op in self.min_st_analysis:
                                min_st = self.min_st_analysis[t_op]
                            if t_dst is not None and (t_dst in (src or [])) and total <= min_st:
                                up_ok = False
                                break

        # ---------- 下移检查 ----------
        if down_ok:
            nxt = kernel_section[lineno + 1].strip()
            n_ctrl, _, n_pred, n_op, n_dst, n_src, n_meta = decode(nxt)
            if n_ctrl is None:
                down_ok = False
            # 直接依赖
            elif (dst in (n_src or [])) or (n_dst in (src or [])):
                down_ok = False
            # 语义邻接规则
            elif not check_adj_opcodes(
                self.cc,
                op or "",
                n_op or "",
                dst,
                n_dst,
                src or [],
                n_src or [],
                pred,
                n_pred,
                n_meta,
            ):
                down_ok = False
            else:
                # 邻居禁令
                for bo in self.ban_ops:
                    if (n_op or "").startswith(bo):
                        down_ok = False
                        break
                # scoreboard（只看 next 的 waits）
                if down_ok:
                    n_waits, *_ = decode_ctrl_code(n_ctrl)
                    if (r in n_waits) or (w in n_waits):
                        down_ok = False

                # 注：原 Sample 对“向下移动的 mem 窗口”是注释掉的，这里保持一致，不做

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
