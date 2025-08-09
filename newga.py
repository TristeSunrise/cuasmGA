from dataclasses import dataclass
from collections import defaultdict, Counter
import random
from typing import List, Dict, Set, Tuple, Optional, Callable
from collections import Counter
import random
from typing import Callable, Optional
from sass_kernel import SassKernel
from sassgen import write_sass_file


POP_SIZE = 10   #population size
MUTATION_RATE = 1
NUM_GENERATIONS = 10
ELITE_SIZE = 4
# --- 把文本序列 <-> 基线ID序列 的映射（处理重复指令文本） ---
def _make_catalog(baseline: List[str]) -> Dict[str, List[int]]:
    cat: Dict[str, List[int]] = {}
    for idx, line in enumerate(baseline):
        cat.setdefault(line, []).append(idx)
    return cat

def _to_ids(seq: List[str], baseline: List[str], catalog=None) -> List[int]:
    if catalog is None:
        catalog = _make_catalog(baseline)
    counters = defaultdict(int)
    ids = []
    for s in seq:
        k = counters[s]
        if s not in catalog or k >= len(catalog[s]):
            raise ValueError(f"发现与基线不一致的指令文本：{s!r}")
        ids.append(catalog[s][k])
        counters[s] += 1
    # print(f"ids: {len(ids)}")
    return ids

def _to_lines(ids: List[int], baseline: List[str]) -> List[str]:
    return [baseline[i] for i in ids]

# --- 拓扑合法性检查（保险） ---
def _is_topological(order: List[int], preds: Dict[int, Set[int]]) -> bool:
    pos = {v: i for i, v in enumerate(order)}
    for v, ps in preds.items():
        pv = pos[v]
        for u in ps:
            if pos[u] >= pv:
                return False
    return True

# --- 列表调度（就绪集合里按 key 选择），把 “随机键/优先级” 解码为拓扑序 ---
def _list_schedule_by_keys(keys: List[float], preds: Dict[int, Set[int]]) -> List[int]:
    N = len(keys)
    indeg = [0]*N
    succs = defaultdict(list)
    for v, ps in preds.items():
        indeg[v] = len(ps)
        for u in ps:
            succs[u].append(v)

    ready = {i for i in range(N) if indeg[i] == 0}
    taken = [False]*N
    order: List[int] = []

    while len(order) < N:
        if not ready:
            raise RuntimeError("DAG 断言失败：ready 为空。")
        # 选 key 最小的就绪点；平手用索引打破（稳定）
        v = min(ready, key=lambda i: (keys[i], i))
        order.append(v)
        taken[v] = True
        ready.remove(v)
        for w in succs[v]:
            indeg[w] -= 1
            if indeg[w] == 0 and not taken[w]:
                ready.add(w)
    return order

# --- PPX（先序保持）交叉：合并两个合法拓扑序，产出仍合法拓扑序 ---
def _ppx_ids(p1: List[int], p2: List[int], preds: Dict[int, Set[int]], rng=None) -> List[int]:
    if rng is None:
        rng = random.Random()
    N = len(p1)
    pos1 = {v:i for i,v in enumerate(p1)}
    pos2 = {v:i for i,v in enumerate(p2)}

    indeg = [0]*N
    succs = defaultdict(list)
    for v, ps in preds.items():
        indeg[v] = len(ps)
        for u in ps:
            succs[u].append(v)

    ready = {i for i in range(N) if indeg[i] == 0}
    taken = [False]*N
    child: List[int] = []
    i1 = i2 = 0

    def next_ready(seq, start):
        j = start
        while j < N:
            v = seq[j]
            if (not taken[v]) and (v in ready):
                return j, v
            j += 1
        return None

    while len(child) < N:
        c1 = next_ready(p1, i1)
        c2 = next_ready(p2, i2)

        if c1 and c2:
            _, a = c1; _, b = c2
            if a == b:
                chosen = a
            else:
                # 谁在“另一个父代”更靠前，就选谁；平手随机
                if pos2[a] < pos1[b]:
                    chosen = a
                elif pos1[b] < pos2[a]:
                    chosen = b
                else:
                    chosen = a if rng.random() < 0.5 else b
        elif c1:
            _, chosen = c1
        elif c2:
            _, chosen = c2
        else:
            # 都取不到时，从 ready 里选 “pos1+pos2 最小”的
            chosen = min(ready, key=lambda v: pos1[v] + pos2[v])

        child.append(chosen)
        taken[chosen] = True
        if chosen in ready:
            ready.remove(chosen)
        for w in succs[chosen]:
            indeg[w] -= 1
            if indeg[w] == 0 and not taken[w]:
                ready.add(w)
        while i1 < N and taken[p1[i1]]: i1 += 1
        while i2 < N and taken[p2[i2]]: i2 += 1

    assert _is_topological(child, preds)
    return child

# --- 用“随机键+轻微扰动”的安全变异（重新列表调度，始终合法） ---
def _mutate_ids_by_key_jitter(ids: List[int], preds: Dict[int, Set[int]], strength: float = 0.05, rng=None) -> List[int]:
    if rng is None:
        rng = random.Random()
    N = len(ids)
    # 以当前顺序当作基础键（位置即键），加入微小噪声
    base_keys = [0.0]*N
    for pos, v in enumerate(ids):
        base_keys[v] = float(pos)
    # 随机挑一些点加/减小扰动
    k = max(1, N // 50)  # 约 2% 的节点扰动；可调
    chosen = rng.sample(range(N), k)
    for v in chosen:
        base_keys[v] += rng.uniform(-strength, strength) * N
    # 重新解码为合法拓扑序
    return _list_schedule_by_keys(base_keys, preds)
def _assert_perm(ids: list[int], N: int):
    if len(ids) != N or set(ids) != set(range(N)):
        raise AssertionError("ids 不是 [0..N-1] 的排列")

def _check_multiset(lines: list[str], baseline_counter: Counter):
    assert Counter(lines) == baseline_counter, "指令多重集不一致"

def _build_succs(preds):
    from collections import defaultdict
    succs = defaultdict(list)
    for v, ps in preds.items():
        for u in ps:
            succs[u].append(v)
    return succs

def _topo_order(preds):
    from collections import deque
    N = len(preds)
    indeg = [0]*N
    succs = _build_succs(preds)
    for v, ps in preds.items():
        indeg[v] = len(ps)
    q = deque([i for i in range(N) if indeg[i]==0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for w in succs[v]:
            indeg[w] -= 1
            if indeg[w]==0:
                q.append(w)
    if len(order)!=N:
        raise RuntimeError("Not a DAG")
    return order

def _precompute_ancestors(preds):
    """anc[v] = 所有必须在 v 之前的节点（传递闭包）"""
    N = len(preds)
    order = _topo_order(preds)
    anc = [set() for _ in range(N)]
    for v in order:
        S = set()
        for u in preds[v]:
            S.add(u); S |= anc[u]
        anc[v] = S
    return anc

# ======== 下面是你原 GA 的“最小改造版本” ========

class Individual:
    def __init__(self, kernel_section: List[str]):
        self.sass = kernel_section
        self.fitness: Optional[float] = None
        

class GeneticAlgorithm:
    original_kernel_section: Optional[list] = None

    def __init__(self, kernel_section: List[str],
                 sasskernel: SassKernel,
                 movable_mask,
                 test_correctness,
                 test_performance: Callable[[Individual], float],
                 preds: Dict[int, Set[int]]):
        # 基线 & DAG
        self.original_kernel_section = kernel_section
        self.sasskernel = sasskernel
        self.baseline = kernel_section[:]            # 用作全集/映射基准
        self.preds = preds
        self.catalog = _make_catalog(self.baseline)
        self.movable_mask = movable_mask or [True] * len(self.baseline)  # 无则全开
        assert len(self.movable_mask) == len(self.baseline)
        self.anc = _precompute_ancestors(self.preds)

        self.counter = Counter(kernel_section)
        print(f"Existing duplicate？：{any(c>1 for c in self.counter.values())}")

        self.test_correctness = test_correctness
        self.test_performance = test_performance

        # 预缓存“基线ID顺序”，便于编码/解码
        self.baseline_ids = list(range(len(self.baseline)))

    def _to_ids(self, seq: List[str]) -> List[int]:
        return _to_ids(seq, self.baseline, self.catalog)

    def _to_lines(self, ids: List[int]) -> List[str]:
        return _to_lines(ids, self.baseline)

    def evaluate_fitness(self, individual: Individual) -> float:
        # 可选：先做 correctness gate（强烈建议）
        try:
            updated_sass = self.sasskernel._update_kernel(individual.sass)
            ok = self.test_correctness(write_sass_file(updated_sass))
            if not ok:
                individual.fitness = float("inf")
            else:
                f = self.test_performance(individual)
                individual.fitness = float(f) if f is not None else float("inf")
        except Exception:
            individual.fitness = float("inf")
        return individual.fitness

    # ---- 初始化：用 “随机键 + 列表调度” 采样拓扑序（而不是 random.shuffle）----
    def initialize_population(self, original_kernel_section: List[str]):
        population = []
        N = len(original_kernel_section)

        # (a) 基线个体（保证至少有一个可运行）
        base_ids = list(range(N))
        base_sass = self._to_lines(base_ids)
        base_ind = Individual(base_sass)
        base_ind.fitness = self.evaluate_fitness(base_ind)
        population.append(base_ind)

        # (b) 其余：keys=基线位置 + 小噪声（只对 ALU 节点加噪）
        tries = 0
        while len(population) < POP_SIZE and tries < POP_SIZE * 20:
            tries += 1
            keys = [float(i) for i in range(N)]
            for i in range(N):
                if self.movable_mask[i]:
                    # 轻微扰动，别太大（严格 preds 下 0.1~0.3 足够）
                    keys[i] += random.uniform(-0.2, 0.2)
            try:
                ids = _list_schedule_by_keys(keys, self.preds)
                sass = self._to_lines(ids)
                if Counter(sass) != self.counter:
                    continue
                ind = Individual(sass)
                ind.fitness = self.evaluate_fitness(ind)
                if ind.fitness != float("inf"):
                    population.append(ind)
            except Exception:
                # 生成/评估失败就重试
                continue

        # 若仍不足，补基线拷贝，保证稳定启动
        while len(population) < POP_SIZE:
            dup = Individual(base_sass[:])
            dup.fitness = base_ind.fitness
            population.append(dup)

        return population
    
    # ---- 交叉：PPX，保证子代仍是 DAG 的拓扑序 ----
    # def crossover(self, parent1: Individual, parent2: Individual):
    #     """
    #     rank-mix crossover：
    #     - 用两个父代的名次 rank1/rank2 混合得到 key
    #     - 仅对 movable 节点加小噪声
    #     - 用列表调度按 DAG 解码得到合法子代
    #     - 若子代与父代完全一样，做一次很小的 jitter 保证差异
    #     """
    #     # --- 把父代转成 ID 序列 ---
    #     p1 = self._to_ids(parent1.sass)
    #     p2 = self._to_ids(parent2.sass)
    #     N  = len(p1)
    #     movable = getattr(self, "movable_mask", None) or [True] * N

    #     def _rank(order: List[int]) -> List[int]:
    #         r = [0] * N
    #         for i, v in enumerate(order):
    #             r[v] = i
    #         return r

    #     def _mix_child(alpha: float, noise: float) -> List[int]:
    #         r1, r2 = _rank(p1), _rank(p2)
    #         keys = [0.0] * N
    #         for v in range(N):
    #             base = alpha * r1[v] + (1.0 - alpha) * r2[v]
    #             if movable[v]:
    #                 base += random.uniform(-noise, noise) * N  # 仅对可移动节点加噪
    #             keys[v] = base
    #         return _list_schedule_by_keys(keys, self.preds)

    #     try:
    #         # 生成两种风味的子代（父代权重不同）
    #         child_ids_1 = _mix_child(alpha=0.35, noise=0.12)
    #         child_ids_2 = _mix_child(alpha=0.65, noise=0.12)
    #     except Exception:
    #         # 兜底：回退为父代拷贝，且确保有 fitness
    #         c1 = Individual(parent1.sass[:])
    #         c2 = Individual(parent2.sass[:])
    #         c1.fitness = parent1.fitness if parent1.fitness is not None else self.evaluate_fitness(c1)
    #         c2.fitness = parent2.fitness if parent2.fitness is not None else self.evaluate_fitness(c2)
    #         return c1, c2

    #     # 若子代与对应父代完全一致，做一次很小的 jitter，保证“有变化但合法”
    #     if child_ids_1 == p1:
    #         try:
    #             child_ids_1 = _mutate_ids_by_key_jitter(child_ids_1, self.preds, strength=0.03)
    #         except Exception:
    #             pass
    #     if child_ids_2 == p2:
    #         try:
    #             child_ids_2 = _mutate_ids_by_key_jitter(child_ids_2, self.preds, strength=0.03)
    #         except Exception:
    #             pass

    #     # --- 映射回文本，并做多重集一致性校验（防止意外） ---
    #     c1_lines = self._to_lines(child_ids_1)
    #     if Counter(c1_lines) != self.counter:
    #         c1_lines = parent1.sass[:]  # 兜底回退
    #     c2_lines = self._to_lines(child_ids_2)
    #     if Counter(c2_lines) != self.counter:
    #         c2_lines = parent2.sass[:]

    #     # --- 构造个体并评估 ---
    #     c1 = Individual(c1_lines); c1.fitness = self.evaluate_fitness(c1)
    #     c2 = Individual(c2_lines); c2.fitness = self.evaluate_fitness(c2)
    #     return c1, c2
    def crossover(self, parent1: Individual, parent2: Individual):
        c1 = Individual(parent1.sass[:]); c1.fitness = parent1.fitness if parent1.fitness is not None else self.evaluate_fitness(c1)
        c2 = Individual(parent2.sass[:]); c2.fitness = parent2.fitness if parent2.fitness is not None else self.evaluate_fitness(c2)
        return c1, c2


    # ---- 变异：对当前顺序做“键扰动→重调度”，始终合法 ----
    def mutate(self, individual: Individual, max_steps: int = 2) -> Individual:
        # 纯变异版本：最小幅度移动计算指令（不破坏依赖）
        if random.random() >= MUTATION_RATE:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        ids = self._to_ids(individual.sass)
        N = len(ids)
        pos = {v:i for i,v in enumerate(ids)}

        # 候选：可移动掩码（建议是 ALU 且无谓词/无 CC/无内存副作用）
        movable = [i for i in range(N) if self.movable_mask[i]]
        if not movable:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        v = random.choice(movable)
        steps = random.randint(1, max_steps)
        direction = random.choice([-1, +1])  # -1 向前，+1 向后

        moved = False
        for _ in range(steps):
            i = pos[v]
            j = i + direction
            if not (0 <= j < N):
                break
            w = ids[j]
            # 关键：判断跨过相邻 w 是否仍拓扑合法
            # 向前移动 v（把 v 放到 w 前） => 不能让 v 早于它的任一祖先，尤其是 w 若是祖先就不行
            if direction < 0:
                if w in self.anc[v]:
                    break
            else:
                # 向后移动 v（把 v 放到 w 后） => 不能让 v 晚于它的任一后继；等价检查 v 是否是 w 的祖先
                if v in self.anc[w]:
                    break
            # 交换相邻（安全）
            ids[i], ids[j] = ids[j], ids[i]
            pos[v], pos[w] = j, i
            moved = True

        if not moved:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        # 映射回文本并评估
        individual.sass = self._to_lines(ids)
        individual.fitness = self.evaluate_fitness(individual)
        return individual



    # ---- 你的 run_ga 逻辑基本不变，仅初始化已换成合法拓扑采样 ----
    def run_ga(self, originol_pure_kernel: List[str]):
        origin = Individual(originol_pure_kernel)
        print("testing the correctness of the original kernel")
        origin.fitness = self.evaluate_fitness(origin)
        print(f"original kernel fitness: {origin.fitness}")
        population = self.initialize_population(originol_pure_kernel)

        for gen in range(NUM_GENERATIONS):
            best = min(population, key=lambda x: x.fitness)
            print(f"GEN {gen} best fitness: {best.fitness}")

            population.sort(key=lambda x: x.fitness)
            next_gen = population[:ELITE_SIZE]

            while len(next_gen) < POP_SIZE:
                parent1 = self.tournament_selection(population, k=4)
                parent2 = self.tournament_selection(population, k=4)
                child1, child2 = self.crossover(parent1, parent2)
                if child1 and child1.fitness != float("inf"):
                    next_gen.append(self.mutate(child1))
                if len(next_gen) < POP_SIZE and child2 and child2.fitness != float("inf"):
                    next_gen.append(self.mutate(child2))
            population = next_gen

        best = min(population, key=lambda x: x.fitness)
        return best

    # 你原来的 tournament_selection 保留
    def tournament_selection(self, population, k=4):
        contenders = random.sample(population, k)
        return min(contenders, key=lambda x: x.fitness)
